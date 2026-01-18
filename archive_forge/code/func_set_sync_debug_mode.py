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
def set_sync_debug_mode(debug_mode: Union[int, str]) -> None:
    """Set the debug mode for cuda synchronizing operations.

    Args:
        debug_mode(str or int): if "default" or 0, don't error or warn on synchronizing operations,
            if "warn" or 1, warn on synchronizing operations, if "error" or 2, error out synchronizing operations.

    Warning:
        This is an experimental feature, and not all synchronizing operations will trigger warning or error. In
        particular, operations in torch.distributed and torch.sparse namespaces are not covered yet.
    """
    _lazy_init()
    if isinstance(debug_mode, str):
        if debug_mode == 'default':
            debug_mode = 0
        elif debug_mode == 'warn':
            debug_mode = 1
        elif debug_mode == 'error':
            debug_mode = 2
        else:
            raise RuntimeError('invalid value of debug_mode, expected one of `default`, `warn`, `error`')
    torch._C._cuda_set_sync_debug_mode(debug_mode)