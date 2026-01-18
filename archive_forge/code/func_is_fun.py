import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import definitely_true, sym_eq
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from .schemas import MutationType
def is_fun(t):
    if isinstance(t, Tensor) and is_traceable_wrapper_subclass(t):
        t_attrs, _ = t.__tensor_flatten__()
        t_inners = [getattr(t, attr) for attr in t_attrs]
        any_fun = any((is_fun(x) for x in t_inners))
        all_fun = all((is_fun(x) for x in t_inners))
        assert any_fun == all_fun
        return any_fun
    return isinstance(t, FunctionalTensor)