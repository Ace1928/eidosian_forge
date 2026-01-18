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
def diff_meta(tensor_vars1, tensor_vars2):
    assert all((isinstance(var, TensorVariable) for var in tensor_vars1 + tensor_vars2))
    all_diffs = []
    for i, (var1, var2) in enumerate(zip(tensor_vars1, tensor_vars2)):
        meta1 = _extract_tensor_metadata(var1.proxy.node.meta['example_value'])
        meta2 = _extract_tensor_metadata(var2.proxy.node.meta['example_value'])
        if meta1 != meta2:
            all_diffs.append((f'pair{i}:', meta1, meta2))
    return all_diffs