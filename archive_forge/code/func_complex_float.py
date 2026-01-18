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
def complex_float(self):
    """Casts this storage to complex float type."""
    _warn_typed_storage_removal()
    return self._to(torch.cfloat)