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
def remove_unused_graphargs(self) -> None:
    assert self.should_exit
    for node in reversed(list(self.graph.nodes)):
        if len(list(node.users)) == 0:
            if node.op == 'get_attr':
                self.remove_node(node)
            elif node.op == 'call_function' and node.target is operator.getitem:
                self.remove_node(node)

    def placeholder_binds_symbol(node):
        arg = node.meta['grapharg']
        example = arg.example
        if isinstance(example, torch.SymInt) and isinstance(example.node.expr, sympy.Symbol):
            return example.node.expr
        return None

    def remove_unused(node):
        log.debug('REMOVE UNUSED GRAPHARG %s', node.meta['grapharg'].source.name())
        del node.meta['grapharg']
        self.remove_node(node)
        self.real_value_cache.pop(node, None)
    used_symbols = set()
    recheck_placeholders = []
    for node in self.placeholders:
        binds_symbol = placeholder_binds_symbol(node) is not None
        if binds_symbol:
            if not node.users:
                recheck_placeholders.append(node)
        elif not node.users:
            remove_unused(node)
        else:
            arg = node.meta['grapharg']
            fake = arg.fake_tensor if arg.fake_tensor is not None else arg.example
            used_symbols |= free_symbols(fake)
    for node in recheck_placeholders:
        symbol = placeholder_binds_symbol(node)
        if symbol is not None:
            if symbol not in used_symbols:
                remove_unused(node)
            else:
                used_symbols.remove(symbol)