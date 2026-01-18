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
@_exception_logger
def scatter_object_list(scatter_object_output_list, scatter_object_input_list, src=0, group=None):
    """
    Scatters picklable objects in ``scatter_object_input_list`` to the whole group.

    Similar to :func:`scatter`, but Python objects can be passed in. On
    each rank, the scattered object will be stored as the first element of
    ``scatter_object_output_list``. Note that all objects in
    ``scatter_object_input_list`` must be picklable in order to be scattered.

    Args:
        scatter_object_output_list (List[Any]): Non-empty list whose first
            element will store the object scattered to this rank.
        scatter_object_input_list (List[Any]): List of input objects to scatter.
            Each object must be picklable. Only objects on the ``src`` rank will
            be scattered, and the argument can be ``None`` for non-src ranks.
        src (int): Source rank from which to scatter
            ``scatter_object_input_list``.
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        ``None``. If rank is part of the group, ``scatter_object_output_list``
        will have its first element set to the scattered object for this rank.

    .. note:: Note that this API differs slightly from the scatter collective
        since it does not provide an ``async_op`` handle and thus will be a
        blocking call.

    .. warning::
        :func:`scatter_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`scatter_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`scatter` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     # Can be any list on non-src ranks, elements are not used.
        >>>     objects = [None, None, None]
        >>> output_list = [None]
        >>> dist.scatter_object_list(output_list, objects, src=0)
        >>> # Rank i gets objects[i]. For example, on rank 2:
        >>> output_list
        [{1: 2}]
    """
    if _rank_not_in_group(group):
        _warn_not_in_group('scatter_object_list')
        return
    if not isinstance(scatter_object_output_list, list) or len(scatter_object_output_list) < 1:
        raise ValueError('Expected argument scatter_object_output_list to be a list of size at least 1.')
    my_rank = get_rank()
    pg_device = _get_pg_default_device(group)
    if my_rank == src:
        tensor_list, tensor_sizes = zip(*[_object_to_tensor(obj, pg_device) for obj in scatter_object_input_list])
        tensor_list, tensor_sizes = (list(tensor_list), list(tensor_sizes))
    if my_rank == src:
        max_tensor_size = max(tensor_sizes)
        for tensor in tensor_list:
            tensor.resize_(max_tensor_size)
    else:
        max_tensor_size = torch.tensor([0], dtype=torch.long, device=pg_device)
    broadcast(max_tensor_size, src=src, group=group)
    output_tensor = torch.empty(max_tensor_size.item(), dtype=torch.uint8, device=pg_device)
    scatter(output_tensor, scatter_list=None if my_rank != src else tensor_list, src=src, group=group)
    obj_tensor_size = torch.tensor([0], dtype=torch.long, device=pg_device)
    scatter(obj_tensor_size, scatter_list=None if my_rank != src else tensor_sizes, src=src, group=group)
    scatter_object_output_list[0] = _tensor_to_object(output_tensor, obj_tensor_size)