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
def get_process_group_ranks(group: ProcessGroup):
    """
    Get all ranks associated with ``group``.

    Args:
        group (ProcessGroup): ProcessGroup to get all ranks from.

    Returns:
        List of global ranks ordered by group rank.
    """
    return list(_world.pg_group_ranks[group].keys())