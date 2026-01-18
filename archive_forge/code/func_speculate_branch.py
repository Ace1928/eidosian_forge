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
def speculate_branch(branch):
    ix = 1 if branch else 2
    (ret_val, ret_treespec), ret_graph, ret_lifted_freevars = speculate_subgraph(tx, args[ix], operands, {}, graph_checkpoint, checkpoint, 'cond', source_target=self.value, manually_set_subgraph_inputs=False, should_flatten_outputs=True)
    if not only_consist_of(ret_val, (TensorVariable,)):
        unimplemented('Expected branches to return a possibly nested list/tuple/dict of tensors but it consists of non tensors.')
    return (ret_val, ret_treespec, ret_graph, ret_lifted_freevars)