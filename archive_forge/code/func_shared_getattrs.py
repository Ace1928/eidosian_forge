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
def shared_getattrs(true_lifted_proxies, false_lifted_proxies):
    true_targets = {proxy.node.target: proxy for proxy in true_lifted_proxies if proxy.node.op == 'get_attr'}
    true_fn_shared_getattrs = {}
    false_fn_shared_getattrs = {}
    for false_proxy in false_lifted_proxies:
        if false_proxy.node.op == 'get_attr' and false_proxy.node.target in true_targets:
            true_proxy = true_targets[false_proxy.node.target]
            true_fn_shared_getattrs[true_proxy] = true_proxy
            false_fn_shared_getattrs[false_proxy] = true_proxy
    return (true_fn_shared_getattrs, false_fn_shared_getattrs)