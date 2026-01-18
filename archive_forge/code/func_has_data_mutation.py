import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import definitely_true, sym_eq
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from .schemas import MutationType
def has_data_mutation(t):
    if is_traceable_wrapper_subclass(t):
        attrs, _ = t.__tensor_flatten__()
        return any((has_data_mutation(getattr(t, attr)) for attr in attrs))
    else:
        if isinstance(t, torch.Tensor):
            assert isinstance(t, FunctionalTensor)
            return torch._functionalize_has_data_mutation(t.elem)
        return False