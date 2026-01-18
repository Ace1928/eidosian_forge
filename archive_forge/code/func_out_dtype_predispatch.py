import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import (
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._higher_order_ops.utils import autograd_not_implemented
@out_dtype.py_impl(DispatchKey.PreDispatch)
def out_dtype_predispatch(*args, **kwargs):
    with torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(DispatchKey.PreDispatch)):
        return out_dtype(*args, **kwargs)