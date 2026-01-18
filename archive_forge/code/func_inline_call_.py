import collections
import contextlib
import copy
import dataclasses
import dis
import functools
import importlib
import inspect
import itertools
import linecache
import logging
import operator
import sys
import textwrap
import threading
import traceback
import types
import typing
import weakref
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Type
from unittest.mock import patch
import torch
import torch._logging
from torch._guards import Checkpointable, tracing, TracingContext
from . import (
from .allowed_functions import is_allowed, is_builtin_constant, is_forbidden
from .bytecode_analysis import (
from .bytecode_transformation import (
from .code_context import code_context
from .codegen import PyCodegen
from .current_scope_id import current_scope_id
from .exc import ArgsMismatchError, BackendCompilerFailed, unimplemented, Unsupported
from .funcname_cache import get_funcname
from .guards import GuardBuilder, install_guard
from .output_graph import GraphCompileReason, OutputGraph, OutputGraphState
from .replay_record import DummyModule, ExecutionRecorder
from .resume_execution import ContinueExecutionCache, ReenterWith
from .source import (
from .utils import (
from .variables.base import (
from .variables.builder import VariableBuilder, wrap_fx_proxy
from .variables.builtin import BuiltinVariable
from .variables.constant import ConstantVariable, EnumVariable
from .variables.ctx_manager import (
from .variables.dicts import ConstDictVariable, SetVariable
from .variables.functions import (
from .variables.lists import (
from .variables.misc import (
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
from .variables.torch import TorchVariable
from .variables.user_defined import (
@staticmethod
def inline_call_(parent, func: VariableTracker, args: List[VariableTracker], kwargs):
    assert isinstance(func, (UserFunctionVariable, NestedUserFunctionVariable))
    result = InliningInstructionTranslator.check_inlineable(func)
    assert result.skipped is False
    try:
        sub_locals, closure_cells = func.bind_args(parent, args, kwargs)
    except TypeError as e:
        raise ArgsMismatchError('{reason}.\n  func = {func}, args = {args}, kwargs = {kwargs}'.format(reason=str(e), func=f"'{func.get_name()}' {func.get_filename()}:{func.get_code().co_firstlineno}", args=[arg.python_type() for arg in args], kwargs=kwargs))
    for v in itertools.chain(sub_locals.values(), closure_cells.values()):
        if not isinstance(v, VariableTracker):
            unimplemented(f'unconverted arg {v}')
    code: types.CodeType = func.get_code()
    if code.co_name in ('__setitem__', '__setattr__') and (not (args is not None and len(args) > 0 and isinstance(args[0], variables.CustomizedDictVariable))):
        unimplemented(f'inline {code.co_name}')
    suffix = ''
    if torch._logging._internal.log_state.is_artifact_enabled('output_code'):
        suffix = f'\n{dis.Bytecode(code).dis()}'
    if sys.version_info >= (3, 11):
        cur_inst = parent.current_instruction
        parent_code = parent.f_code
        header = parent.get_line_of_code_header(lineno=cur_inst.positions.lineno)

        def get_trace_call_log_str():
            line = get_instruction_source_311(parent_code, cur_inst).rstrip()
            return f'TRACE inlined call {code.co_name} from {header}\n{line}'
        trace_call_log.debug('%s', LazyString(get_trace_call_log_str))
    log.debug('INLINING %s%s, %s', code, suffix, result.reason)
    if args and isinstance(args[0], NNModuleVariable):
        module = parent.output.get_submodule(args[0].module_key)
        if isinstance(module, torch.fx.GraphModule):
            code_context.get_context(module.forward.__code__)['orig_graphmodule'] = module
    tracer: InliningInstructionTranslator
    if is_generator(code):
        tracer = InliningGeneratorInstructionTranslator(parent, code, sub_locals, parent.symbolic_globals, closure_cells, func)
    else:
        tracer = InliningInstructionTranslator(parent, code, sub_locals, parent.symbolic_globals, closure_cells, func)
    strict_ctx: Any = contextlib.nullcontext()
    if parent.strict_checks_enabled:
        strict_ctx = tracer.strict_translation_mode()
    try:
        with strict_ctx:
            tracer.run()
    except exc.SkipFrame as e:
        msg = f'SKIPPED INLINING {code}: {e}'
        log.debug(msg)
        raise Unsupported(msg) from e
    except Exception as e:
        log.debug('FAILED INLINING %s', code)
        raise
    assert tracer.symbolic_result is not None
    func.export_freevars(parent, tracer)
    if tracer.f_globals is parent.f_globals:
        parent.symbolic_globals.update(tracer.symbolic_globals)
    parent.inconsistent_side_effects |= tracer.inconsistent_side_effects
    log.debug('DONE INLINING %s', code)
    if is_generator(code):
        assert isinstance(tracer, InliningGeneratorInstructionTranslator)
        assert tracer.symbolic_result.as_python_constant() is None
        return ListIteratorVariable(tracer.generated_items, mutable_local=MutableLocal())
    else:
        return tracer.symbolic_result