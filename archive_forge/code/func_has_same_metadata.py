import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import definitely_true, sym_eq
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from .schemas import MutationType
def has_same_metadata(t1, t2):
    return definitely_true(sym_eq(t1.size(), t2.size())) and definitely_true(sym_eq(t1.stride(), t2.stride())) and definitely_true(t1.storage_offset() == t2.storage_offset()) and (t1.is_conj() == t2.is_conj()) and (t1.is_neg() == t2.is_neg())