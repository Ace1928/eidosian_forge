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
def supports_complex(reduceOp: ReduceOp) -> bool:
    """Return true if reduce ops is supported. False otherwise."""
    denyList = [ReduceOp.MAX, ReduceOp.MIN, ReduceOp.PRODUCT, ReduceOp.BAND, ReduceOp.BOR, ReduceOp.BXOR]
    return reduceOp not in denyList