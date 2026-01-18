import multiprocessing
import os
import threading
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Union
import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
def reduce_storage(storage):
    from . import get_sharing_strategy
    if storage.is_cuda:
        raise RuntimeError('Cannot pickle CUDA storage; try pickling a CUDA tensor instead')
    elif get_sharing_strategy() == 'file_system':
        metadata = storage._share_filename_cpu_()
        cache_key = metadata[1]
        rebuild = rebuild_storage_filename
        if isinstance(storage, torch.TypedStorage):
            metadata += (storage.dtype,)
        storage._shared_incref()
    elif storage.size() == 0:
        return (rebuild_storage_empty, (type(storage),))
    else:
        fd, size = storage._share_fd_cpu_()
        df = multiprocessing.reduction.DupFd(fd)
        cache_key = fd_id(fd)
        metadata = (df, size)
        rebuild = rebuild_storage_fd
    shared_cache[cache_key] = StorageWeakRef(storage)
    return (rebuild, (type(storage),) + metadata)