import itertools
import collections.abc
import contextlib
import hashlib
import io
import logging
import os
import pickle
import sys
import time
import warnings
from collections import namedtuple
from datetime import timedelta
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
import torch
from torch._C._distributed_c10d import (
from .constants import default_pg_timeout, default_pg_nccl_timeout
from .c10d_logger import _exception_logger, _time_logger
from .rendezvous import register_rendezvous_handler, rendezvous  # noqa: F401
def get_backend_config(group: Optional[ProcessGroup]=None) -> str:
    """
    Return the backend configuration of the given process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        The backend configuration of the given process group as a lower case string.

    """
    if group is None:
        pg = _get_default_group()
    else:
        pg = group
    if _rank_not_in_group(pg):
        raise ValueError('Invalid process group specified')
    backend_config = _world.pg_backend_config.get(pg)
    assert backend_config is not None
    return str(backend_config)