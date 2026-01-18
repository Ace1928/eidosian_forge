import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import definitely_true, sym_eq
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from .schemas import MutationType
def sync_functional_tensor(t):
    if is_traceable_wrapper_subclass(t):
        attrs, ctx = t.__tensor_flatten__()
        for attr in attrs:
            sync_functional_tensor(getattr(t, attr))
    else:
        torch._sync(t)