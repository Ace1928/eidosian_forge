import collections
import contextlib
import copy
import functools
import itertools
import logging
import operator
import re
import sys
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union
import sympy
import torch._guards
import torch._logging
import torch.nn
import torch.utils._pytree as pytree
from torch import fx
from torch._guards import (
from torch._utils_internal import signpost_event
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import free_symbols, is_symbolic, ShapeEnv
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.interp import sympy_interp
from torch.utils._sympy.reference import PythonReferenceAnalysis
from torch.utils.weak import WeakTensorKeyDictionary
from . import config, logging as torchdynamo_logging, variables
from .backends.registry import CompiledFn, CompilerFn
from .bytecode_transformation import (
from .code_context import code_context
from .codegen import PyCodegen
from .current_scope_id import enter_new_scope
from .exc import (
from .guards import GuardBuilder, install_guard
from .mutation_guard import is_dynamic_nn_module
from .side_effects import SideEffects
from .source import (
from .utils import (
from .variables.base import VariableTracker
from .variables.builder import GraphArg, TrackedFake, VariableBuilder, wrap_fx_proxy
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
from .variables.torch_function import TensorWithTFOverrideVariable
def create_graph_input(self, name, type_expr=None, before=False, source=None):
    log.debug('create_graph_input %s %s', name, source.name() if source is not None else '(none)')
    if source is None:
        assert self.parent is not None, 'you are required to provide a source for inputs on the root tracer'
    if self.export_root:
        if not is_from_local_source(source, allow_cell_or_freevar=False):
            self.output_graph.source_to_user_stacks.setdefault(source, []).append(TracingContext.extract_stack())
    if name in self.input_name_to_proxy:
        for i in itertools.count():
            candidate_name = f'{name}_{i}'
            if candidate_name not in self.input_name_to_proxy:
                name = candidate_name
                break
    if self.input_name_to_proxy:
        prev_name = next(reversed(self.input_name_to_proxy))
        node = self.input_name_to_proxy[prev_name].node
        if before:
            ctx = self.graph.inserting_before(node)
        else:
            ctx = self.graph.inserting_after(node)
    else:
        ctx = self.graph.inserting_before(None)
    with ctx:
        proxy = self.create_proxy('placeholder', name, (), {}, type_expr=type_expr)
        if self.input_name_to_proxy and before:
            k, v = self.input_name_to_proxy.popitem()
            self.input_name_to_proxy[name] = proxy
            self.input_name_to_proxy[k] = v
        else:
            self.input_name_to_proxy[name] = proxy
        return proxy