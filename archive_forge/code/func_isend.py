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
def isend(tensor: torch.Tensor, dst: int, group: Optional[ProcessGroup]=None, tag: int=0) -> Work:
    """
    Send a tensor asynchronously.

    .. warning::
        Modifying ``tensor`` before the request completes causes undefined
        behavior.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Args:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with remote recv

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    _check_single_tensor(tensor, 'tensor')
    if _rank_not_in_group(group):
        _warn_not_in_group('isend')
        return
    if group is None or group is GroupMember.WORLD:
        default_pg = _get_default_group()
        return default_pg.send([tensor], dst, tag)
    else:
        group_dst_rank = get_group_rank(group, dst)
        return group.send([tensor], group_dst_rank, tag)