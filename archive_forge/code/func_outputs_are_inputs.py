import functools
import warnings
from typing import Callable, Union
import torch
import torch.utils._pytree as pytree
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import (
from torch.utils._python_dispatch import TorchDispatchMode
def outputs_are_inputs(outputs, inputs):
    input_ids = {id(inp) for inp in tree_flatten_only(torch.Tensor, inputs)}
    return any((id(out) in input_ids for out in tree_flatten_only(torch.Tensor, outputs)))