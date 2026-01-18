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
def custom_function_call_vmap_generate_rule(interpreter, autograd_function, *operands):
    unwrapped_operands, in_dims = unwrap_batched(operands, interpreter.level())
    vmapped_function, get_out_dims = vmapify_autograd_function(autograd_function, in_dims, interpreter.batch_size(), interpreter.randomness())
    with interpreter.lower():
        output = custom_function_call(vmapped_function, *unwrapped_operands)
    out_dims = get_out_dims()
    return wrap_batched(output, out_dims, interpreter.level())