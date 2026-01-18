import torch
from torch._ops import HigherOrderOperator
from torch._C._functorch import TransformType
from torch._functorch.utils import enable_single_level_autograd_function
import torch.utils._pytree as pytree
from torch._C._functorch import (
from torch._functorch.vmap import (
from torch._functorch.apis import vmap
from torch._functorch.vmap import _broadcast_to_and_flatten
from torch.autograd.forward_ad import _set_fwd_grad_enabled
from typing import Any, NamedTuple, Tuple
def reductify_leaf(grad_input, grad_input_bdim, input_bdim, batch_size, target_shape_without_bdim_to_reduce_to=None):
    if grad_input is None:
        return None
    if grad_input_bdim is None and input_bdim is None:
        return grad_input
    if grad_input_bdim is not None and input_bdim is None:
        return grad_input.sum(grad_input_bdim)
    assert input_bdim is not None
    if grad_input_bdim is None:
        grad_input = grad_input.unsqueeze(input_bdim)
        new_shape = list(grad_input.shape)
        new_shape[input_bdim] = batch_size
        grad_input = grad_input.expand(new_shape)
        grad_input_bdim = input_bdim
    if target_shape_without_bdim_to_reduce_to is not None:
        return vmap(torch.Tensor.sum_to_size, in_dims=(grad_input_bdim, None), out_dims=input_bdim)(grad_input, target_shape_without_bdim_to_reduce_to)
    if input_bdim != grad_input_bdim:
        grad_input = grad_input.movedim(grad_input_bdim, input_bdim)
    return grad_input