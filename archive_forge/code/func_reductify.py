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
def reductify(grad_input, grad_input_bdim, input_bdim, batch_size, target_shape_without_bdim_to_reduce_to=None):
    if not isinstance(grad_input, tuple):
        grad_input = (grad_input,)
    if not isinstance(grad_input_bdim, tuple):
        grad_input_bdim = (grad_input_bdim,)
    if not isinstance(input_bdim, tuple):
        input_bdim = (input_bdim,)
    if target_shape_without_bdim_to_reduce_to is None:
        target_shape_without_bdim_to_reduce_to = len(grad_input) * (None,)
    result = tuple((reductify_leaf(gi, gi_bdim, i_bdim, batch_size, maybe_ishape) for gi, gi_bdim, i_bdim, maybe_ishape in zip(grad_input, grad_input_bdim, input_bdim, target_shape_without_bdim_to_reduce_to)))
    return result