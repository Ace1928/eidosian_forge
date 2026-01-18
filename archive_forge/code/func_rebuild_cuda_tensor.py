import multiprocessing
import os
import threading
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Union
import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
def rebuild_cuda_tensor(tensor_cls, tensor_size, tensor_stride, tensor_offset, storage_cls, dtype, storage_device, storage_handle, storage_size_bytes, storage_offset_bytes, requires_grad, ref_counter_handle, ref_counter_offset, event_handle, event_sync_required):
    if storage_handle is None or storage_size_bytes == 0:
        storage = storage_cls(0, dtype=dtype, device=storage_device, _internal=True)
    else:
        storage = storage_from_cache(storage_cls, (storage_handle, storage_offset_bytes))
        if storage is None:
            torch.cuda._lazy_init()
            storage = storage_cls._new_shared_cuda(storage_device, storage_handle, storage_size_bytes, storage_offset_bytes, ref_counter_handle, ref_counter_offset, event_handle, event_sync_required)
            shared_cache[storage_handle, storage_offset_bytes] = StorageWeakRef(storage)
        else:
            storage_cls._release_ipc_counter(ref_counter_handle, ref_counter_offset, device=storage_device)
    _storage = storage if isinstance(storage, torch.UntypedStorage) else storage._untyped_storage
    t = torch._utils._rebuild_tensor(torch.storage.TypedStorage(wrap_storage=_storage, dtype=dtype, _internal=True), tensor_offset, tensor_size, tensor_stride)
    if tensor_cls == torch.nn.parameter.Parameter:
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad
    return t