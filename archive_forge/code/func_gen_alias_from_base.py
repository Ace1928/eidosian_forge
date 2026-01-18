import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import definitely_true, sym_eq
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from .schemas import MutationType
def gen_alias_from_base(aliased_base_tensor, target_meta_tensor, target_requires_grad):
    if target_meta_tensor._base is not None:
        b = target_meta_tensor._base
        abt = aliased_base_tensor
        if abt is not b and (abt.size() != b.size() or abt.stride() != b.stride() or abt.storage_offset() != b.storage_offset()):
            reshaped_base_tensor = aliased_base_tensor.as_strided(b.size(), b.stride(), b.storage_offset())
        else:
            reshaped_base_tensor = aliased_base_tensor
        out = target_meta_tensor._view_func(reshaped_base_tensor)
        if out is not None and out.shape == target_meta_tensor.shape:
            if aliased_base_tensor.requires_grad and (not target_requires_grad):
                out = out.detach()
            elif not aliased_base_tensor.requires_grad and target_requires_grad:
                out.requires_grad_(True)
            return out
    size = target_meta_tensor.size()
    stride = target_meta_tensor.stride()
    storage_offset = target_meta_tensor.storage_offset()
    if aliased_base_tensor.is_complex() and (not target_meta_tensor.is_complex()):
        aliased_out = torch.view_as_real(aliased_base_tensor).as_strided(size, stride, storage_offset)
    elif not aliased_base_tensor.is_complex() and target_meta_tensor.is_complex():
        aliased_out = torch.view_as_complex(aliased_base_tensor).as_strided(size, stride, storage_offset)
    else:
        aliased_out = aliased_base_tensor.as_strided(size, stride, storage_offset)
    if aliased_base_tensor.requires_grad and (not target_requires_grad):
        aliased_out = aliased_out.detach()
    elif not aliased_base_tensor.requires_grad and target_requires_grad:
        aliased_out.requires_grad_(True)
    if aliased_out.dtype != target_meta_tensor.dtype:
        aliased_out = aliased_out.view(target_meta_tensor.dtype)
    return aliased_out