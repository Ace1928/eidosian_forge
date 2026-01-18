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
def break_graph_if_unsupported(*, push):

    def decorator(inner_fn):

        @functools.wraps(inner_fn)
        def wrapper(self: 'InstructionTranslatorBase', inst: Instruction):
            speculation = self.speculate()
            if speculation.failed:
                assert speculation.reason is not None
                return handle_graph_break(self, inst, speculation.reason)
            try:
                TracingContext.set_current_loc(self.f_code.co_filename, self.lineno, self.f_code.co_name)
                return inner_fn(self, inst)
            except Unsupported as excp:
                if self.should_compile_partial_graph() and self.has_backedge():
                    msg = f'Skipping frame because there is a graph break in a for/while loop\n{self.frame_summary()}'
                    log.info(msg)
                    raise exc.SkipFrame(msg) from excp
                if self.generic_context_manager_depth > 0:
                    excp.remove_from_stats()
                    unimplemented('Graph break under GenericContextWrappingVariable')
                if isinstance(excp, exc.UncapturedHigherOrderOpError):
                    raise
                if not self.should_compile_partial_graph():
                    raise
                log.debug('break_graph_if_unsupported triggered compile', exc_info=True)
                user_stack = excp.real_stack
                user_stack_formatted = ''.join(traceback.format_list(user_stack))
                frame_loc = (user_stack[-1].filename, user_stack[-1].lineno)
                if graph_break_log.isEnabledFor(logging.DEBUG) and (not explain) and graph_break_dup_warning_checker.add(frame_loc):
                    graph_break_log.debug('Graph break: %s from user code at:\n%s', excp, user_stack_formatted)
                excp.remove_from_stats()
                excp.add_to_stats('graph_break')
                speculation.reason = GraphCompileReason(excp.msg, user_stack)
            speculation.fail_and_restart_analysis()

        def handle_graph_break(self: 'InstructionTranslatorBase', inst: Instruction, reason: GraphCompileReason):
            self.output.compile_subgraph(self, reason=reason)
            cg = PyCodegen(self)
            cleanup: List[Instruction] = []
            for b in self.block_stack:
                assert b.with_context is not None
                self.output.add_output_instructions([*b.with_context.reconstruct(cg), *b.resume_fn().try_except(cg.code_options, cleanup)])
            if sys.version_info >= (3, 11) and inst.opname == 'CALL':
                kw_names = self.kw_names.as_python_constant() if self.kw_names is not None else ()
                if len(kw_names) > 0:
                    self.output.add_output_instructions([create_instruction('KW_NAMES', argval=kw_names)])
                self.output.add_output_instructions(create_call_function(inst.arg, False))
            else:
                assert inst.target is None
                inst_copy = copy.copy(inst)
                inst_copy.exn_tab_entry = None
                self.output.add_output_instructions([inst_copy])
            self.output.add_output_instructions(cleanup)
            if sys.version_info >= (3, 11) and inst.opname == 'CALL':
                stack_effect = dis.stack_effect(dis.opmap['PRECALL'], inst.arg) + dis.stack_effect(dis.opmap['CALL'], inst.arg)
            else:
                stack_effect = dis.stack_effect(inst.opcode, inst.arg)
            self.popn(push - stack_effect)
            for _ in range(push):
                self.push(UnknownVariable())
            self.output.add_output_instructions(self.create_call_resume_at(self.next_instruction))
        return wrapper
    return decorator