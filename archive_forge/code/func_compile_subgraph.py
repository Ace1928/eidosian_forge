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
def compile_subgraph(self, tx, partial_convert=False, reason: Optional[GraphCompileReason]=None, compile_return_value=False):
    """
        Generate a subgraph to continue execution on user code.
        Automatically restore live variables.
        """
    assert reason is not None
    from .decorators import disable
    self.partial_convert = partial_convert
    self.compile_subgraph_reason = reason
    self.should_exit = True
    if not compile_return_value:
        self.guard_has_graph_break()
    log.debug('COMPILING GRAPH due to %s', reason)
    if not all((block.can_restore() for block in tx.block_stack)):
        unimplemented('compile_subgraph with block_depth != 0')
    prefix_insts: List[Instruction] = []
    if sys.version_info >= (3, 11):
        for inst in tx.prefix_insts:
            if inst.opname == 'MAKE_CELL':
                prefix_insts.append(create_instruction('MAKE_CELL', argval=inst.argval))
            elif inst.opname == 'COPY_FREE_VARS':
                prefix_insts.append(create_instruction('COPY_FREE_VARS', arg=len(tx.code_options['co_freevars'])))
            else:
                prefix_insts.append(copy.copy(inst))

    def append_prefix_insts():
        self.add_output_instructions(prefix_insts)
        prefix_insts.clear()
    for block in reversed(tx.block_stack):
        block.exit(tx)
    self.cleanup_graph()
    tx.prune_dead_locals()
    stack_values = list(tx.stack)
    root = FakeRootModule(self.nn_modules)
    restore_vars = []
    val_to_names: Dict[VariableTracker, List[str]] = {}
    if stack_values:
        val_to_names[stack_values[-1]] = list()
    for k, v in tx.symbolic_locals.items():
        if isinstance(v.source, LocalSource) and v.source.local_name == k:
            continue
        if v not in val_to_names:
            val_to_names[v] = list()
        val_to_names[v].append(k)
    for v in val_to_names.keys():
        restore_vars.extend(val_to_names[v])
        stack_values.extend([v] * len(val_to_names[v]))
    if len(tx.random_calls) > 0:
        append_prefix_insts()
        random_calls_instructions = []
        self.random_values_var = self.new_var('random_values')
        rand_fn_name = unique_id('__gen_rand_values')
        rand_fn = disable(_get_gen_rand_values_fn(tx.random_calls))
        self.install_global(rand_fn_name, rand_fn)
        codegen = PyCodegen(tx, root)
        random_calls_instructions.extend(codegen.load_function_name(rand_fn_name, True))
        random_calls_instructions.extend(create_call_function(0, False))
        random_calls_instructions.append(codegen.create_store(tx.output.random_values_var))
        self.add_output_instructions(random_calls_instructions)
    if stack_values and all((not isinstance(v, (UnspecializedPythonVariable, NumpyNdarrayVariable, TensorWithTFOverrideVariable)) for v in stack_values)) and all((isinstance(x, TensorVariable) for x in stack_values)) and (len(set(stack_values)) == len(stack_values)) and self.side_effects.is_empty():
        append_prefix_insts()
        self.add_output_instructions(self.compile_and_call_fx_graph(tx, list(reversed(stack_values)), root) + [create_instruction('UNPACK_SEQUENCE', arg=len(stack_values))])
    else:
        graph_output_var = self.new_var('graph_out')
        pass1 = PyCodegen(tx, root, graph_output_var)
        self.side_effects.codegen_hooks(pass1)
        self.side_effects.codegen_save_tempvars(pass1)
        pass1.restore_stack(stack_values)
        self.side_effects.codegen_update_mutated(pass1)
        pass2 = PyCodegen(tx, root, graph_output_var, tempvars={val: None for val, count in pass1.uses.items() if count > 1})
        self.side_effects.codegen_hooks(pass2)
        self.side_effects.codegen_save_tempvars(pass2)
        pass2.restore_stack(stack_values)
        self.side_effects.codegen_update_mutated(pass2)
        output = []
        if count_calls(self.graph) != 0 or len(pass2.graph_outputs) != 0:
            output.extend(self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root))
            if len(pass2.graph_outputs) != 0:
                output.append(pass2.create_store(graph_output_var))
            else:
                output.append(create_instruction('POP_TOP'))
        append_prefix_insts()
        self.add_output_instructions(output + pass2.get_instructions())
    self.add_output_instructions([PyCodegen(tx).create_store(var) for var in reversed(restore_vars)])