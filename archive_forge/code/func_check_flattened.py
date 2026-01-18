from contextlib import contextmanager
import torch
import torch._custom_ops
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils import _pytree as pytree
def check_flattened(flat_args):
    for a in flat_args:
        if not (isinstance(a, (torch.Tensor, str, int, float, bool)) or a is None):
            raise AssertionError(f'Only Tensors or scalars are supported as pytree flattened inputs, got: {a}')