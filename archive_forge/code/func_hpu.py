import io
import torch
from ._utils import _type, _cuda, _hpu
from torch.types import Storage
from typing import cast, Any, Dict as _Dict, Optional as _Optional, TypeVar, Type, Union
import copy
import collections
from functools import lru_cache
import warnings
import threading
import functools
def hpu(self, device=None, non_blocking=False, **kwargs) -> T:
    _warn_typed_storage_removal()
    if self.dtype in [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]:
        raise RuntimeError('Cannot create HPU storage with quantized dtype')
    hpu_storage: torch.UntypedStorage = self._untyped_storage.hpu(device, non_blocking, **kwargs)
    return self._new_wrapped_storage(hpu_storage)