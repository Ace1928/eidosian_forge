from typing import Optional, Tuple
import torch
import torch.utils._pytree as pytree
from torch import _prims
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._prims_common import CUDARngStateHelper, make_contiguous_strides_for
from torch._prims_common.wrappers import backwards_not_supported
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
from torch.types import _device, _dtype
@run_with_rng_state.py_impl(DispatchKey.CPU)
def impl_cpu(rng_state, op, *args, **kwargs):
    current_state = torch.get_rng_state()
    torch.set_rng_state(rng_state)
    out = op(*args, **kwargs)
    torch.set_rng_state(current_state)
    return out