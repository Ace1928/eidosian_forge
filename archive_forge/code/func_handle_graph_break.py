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