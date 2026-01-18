import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import (
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._higher_order_ops.utils import autograd_not_implemented
@out_dtype.py_functionalize_impl
def out_dtype_func(ctx, op, output_dtype, *args):
    unwrapped_args = tuple((ctx.unwrap_tensors(arg) for arg in args))
    with ctx.redispatch_to_next():
        res = out_dtype(op, output_dtype, *unwrapped_args)
    return ctx.wrap_tensors(res)