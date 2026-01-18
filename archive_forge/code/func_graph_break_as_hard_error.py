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
@functools.wraps(fn)
def graph_break_as_hard_error(*args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Unsupported as e:
        msg = ' Scroll up to find out what causes the graph break.'
        raise UncapturedHigherOrderOpError(reason + msg) from e