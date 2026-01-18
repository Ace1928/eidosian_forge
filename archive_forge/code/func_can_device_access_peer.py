import contextlib
import importlib
import os
import sys
import threading
import traceback
import warnings
from functools import lru_cache
from typing import Any, cast, List, Optional, Tuple, Union
import torch
import torch._C
from torch.types import Device
from .. import device as _device
from .._utils import classproperty
from ._utils import _dummy_type, _get_device_index
from .graphs import (
from .streams import Event, ExternalStream, Stream
from .memory import *  # noqa: F403
from .random import *  # noqa: F403
from torch.storage import _LegacyStorage, _warn_typed_storage_removal
from . import amp, jiterator, nvtx, profiler, sparse
def can_device_access_peer(device: _device_t, peer_device: _device_t) -> bool:
    """Check if peer access between two devices is possible."""
    _lazy_init()
    device = _get_device_index(device, optional=True)
    peer_device = _get_device_index(peer_device)
    if device < 0 or device >= device_count():
        raise AssertionError('Invalid device id')
    if peer_device < 0 or peer_device >= device_count():
        raise AssertionError('Invalid peer device id')
    return torch._C._cuda_canDeviceAccessPeer(device, peer_device)