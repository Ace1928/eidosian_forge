import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import (
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._higher_order_ops.utils import autograd_not_implemented
def out_dtype_fallback(op, output_dtype, *args):
    flat_inputs = pytree.arg_tree_leaves(*args) + [torch.ones(1, dtype=output_dtype)]
    promote_dtype: torch.dtype = elementwise_dtypes(*flat_inputs, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)[0]
    casted_args = pytree.tree_map_only(torch.Tensor, lambda arg: arg.to(dtype=promote_dtype), args)
    res = op(*casted_args).to(dtype=output_dtype)
    return res