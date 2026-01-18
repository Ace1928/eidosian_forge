import multiprocessing
import os
import threading
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Union
import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
def rebuild_storage_filename(cls, manager, handle, size, dtype=None):
    storage: Union[torch.TypedStorage, torch.UntypedStorage] = storage_from_cache(cls, handle)
    if storage is not None:
        return storage._shared_decref()
    if dtype is None:
        storage = torch.UntypedStorage._new_shared_filename_cpu(manager, handle, size)
    else:
        byte_size = size * torch._utils._element_size(dtype)
        untyped_storage: torch.UntypedStorage = torch.UntypedStorage._new_shared_filename_cpu(manager, handle, byte_size)
        storage = torch.TypedStorage(wrap_storage=untyped_storage, dtype=dtype, _internal=True)
    shared_cache[handle] = StorageWeakRef(storage)
    return storage._shared_decref()