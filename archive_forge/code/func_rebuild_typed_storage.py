import multiprocessing
import os
import threading
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Union
import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
def rebuild_typed_storage(storage, dtype):
    return torch.storage.TypedStorage(wrap_storage=storage, dtype=dtype, _internal=True)