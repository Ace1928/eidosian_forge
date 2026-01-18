import contextlib
import functools
import itertools
import logging
from typing import Dict, List, Optional
import torch._C
import torch.fx
import torch.nn
import torch.onnx.operators
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import deepcopy_to_fake_tensor, get_fake_value, get_real_value
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builtin import BuiltinVariable
from torch._dynamo.variables.functions import UserFunctionVariable
from torch._dynamo.variables.tensor import SymNodeVariable
from torch._guards import Source
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils import _pytree as pytree
from ..exc import (
from ..source import FSDPNNModuleSource, GetItemSource, NNModuleSource
from ..utils import proxy_args_kwargs
from .dicts import ConstDictVariable
from .lists import ListVariable, TupleVariable
from .nn_module import NNModuleVariable, UnspecializedNNModuleVariable
def fixup_branch_inps(graph, lifted_freevars, shared, unique_true, unique_false):

    def _insert_or_replace_phs(new_args, name_suffix):
        for arg in new_args:
            new_ph = graph.placeholder(arg.node.name + name_suffix)
            if arg in lifted_freevars:
                old_ph = lifted_freevars[arg].node
                old_ph.replace_all_uses_with(new_ph)
                old_ph.users = {}
                graph.erase_node(old_ph)
    first_not_ph_node = next((node for node in graph.nodes if node.op != 'placeholder'))
    with graph.inserting_before(first_not_ph_node):
        _insert_or_replace_phs(shared, '')
        _insert_or_replace_phs(unique_true, '_true_branch')
        _insert_or_replace_phs(unique_false, '_false_branch')