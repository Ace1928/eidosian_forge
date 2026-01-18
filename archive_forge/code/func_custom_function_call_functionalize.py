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
@custom_function_call.py_impl(TransformType.Functionalize)
def custom_function_call_functionalize(interpreter, autograd_function, generate_vmap_rule, *operands):
    raise RuntimeError('NYI: Functionalize rule for custom_function_call')