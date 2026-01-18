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
def irecv(tensor: torch.Tensor, src: Optional[int]=None, group: Optional[ProcessGroup]=None, tag: int=0) -> Work:
    """
    Receives a tensor asynchronously.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Args:
        tensor (Tensor): Tensor to fill with received data.
        src (int, optional): Source rank. Will receive from any
            process if unspecified.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match recv with remote send

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    _check_single_tensor(tensor, 'tensor')
    if _rank_not_in_group(group):
        _warn_not_in_group('irecv')
        return
    if group is None or group is GroupMember.WORLD:
        pg = _get_default_group()
    else:
        pg = group
    if src is None:
        return pg.recv_anysource([tensor], tag)
    elif pg is GroupMember.WORLD:
        return pg.recv([tensor], src, tag)
    else:
        group_src_rank = get_group_rank(pg, src)
        return pg.recv([tensor], group_src_rank, tag)