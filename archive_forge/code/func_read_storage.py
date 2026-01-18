import ctypes
import functools
import hashlib
import os.path
import struct
from collections import defaultdict
from typing import Dict, Optional, Set
import torch
import torch._prims as prims
import torch._utils
import torch.nn.functional as F
from torch._C import default_generator
from torch.multiprocessing.reductions import StorageWeakRef
def read_storage(self, h: str, *, device=None) -> torch.UntypedStorage:
    if device is not None:
        device = torch.device(device)
    ws = self.storage_cache[device].get(h) if self.storage_cache is not None else None
    s: Optional[torch.UntypedStorage]
    if ws is not None:
        s = torch.UntypedStorage._new_with_weak_ptr(ws.cdata)
        if s is not None:
            return s
    s = torch.load(os.path.join(self.loc, 'storages', h), weights_only=True, map_location=device)._untyped_storage
    assert s is not None
    if self.storage_cache is not None:
        self.storage_cache[device][h] = StorageWeakRef(s)
    return s