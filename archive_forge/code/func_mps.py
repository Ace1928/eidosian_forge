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
def mps(self):
    """Return a MPS copy of this storage if it's not already on the MPS."""
    if self.device.type != 'mps':
        return torch.UntypedStorage(self.size(), device='mps').copy_(self, False)
    else:
        return self