import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import definitely_true, sym_eq
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from .schemas import MutationType
def has_metadata_mutation(f_arg, arg, *, check_only_storage_mutation: bool):
    if is_traceable_wrapper_subclass(f_arg):
        attrs, _ = f_arg.__tensor_flatten__()
        f_inner_ts = [getattr(f_arg, attr) for attr in attrs]
        inner_ts = [getattr(arg, attr) for attr in attrs]
        return any((has_metadata_mutation(f_inner_t, inner_t, check_only_storage_mutation=check_only_storage_mutation) for f_inner_t, inner_t in zip(f_inner_ts, inner_ts)))
    else:
        if not isinstance(f_arg, torch.Tensor):
            assert not isinstance(arg, torch.Tensor)
            return False
        assert isinstance(f_arg, FunctionalTensor)
        assert isinstance(arg, FakeTensor)
        arg_after = torch._from_functional_tensor(f_arg.elem)
        maybe_storage_changed = torch._functionalize_was_storage_changed(f_arg.elem)
        same_storages = StorageWeakRef(arg.untyped_storage()) == StorageWeakRef(arg_after.untyped_storage())
        has_storage_metadata_mutation = maybe_storage_changed and (not same_storages)
        if check_only_storage_mutation:
            return has_storage_metadata_mutation
        if has_storage_metadata_mutation:
            return True
        maybe_metadata_mutated = torch._functionalize_has_metadata_mutation(f_arg.elem)
        if not maybe_metadata_mutated:
            return False
        same_sizes = arg.shape == arg_after.shape
        same_strides = arg.stride() == arg_after.stride()
        same_offsets = arg.storage_offset() == arg_after.storage_offset()
        has_metadata_mutation_ = maybe_metadata_mutated and (not (same_sizes and same_strides and same_offsets))
        return has_metadata_mutation_