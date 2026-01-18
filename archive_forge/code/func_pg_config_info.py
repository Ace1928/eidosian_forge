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
@property
def pg_config_info(self) -> List[Dict[str, Union[int, str]]]:
    """
        Return a list of dict with process groups and backends.

        Along with their unique IDs and configurations (types and ranks).
        """
    config_info = []
    default_pg_size = _get_group_size(None)
    for pg, backend in self.pg_map.items():
        backend_type = Backend.backend_type_map[backend[0]]
        ranks = self.pg_group_ranks[pg]
        config_info.append({'pg_name': self.pg_names[pg], 'backend_id': pg._backend_id(backend_type), 'backend_config': self.pg_backend_config[pg], 'ranks': list(ranks.keys()) if len(ranks) != default_pg_size else [], 'group_size': len(ranks), 'group_count': self.group_count})
    return config_info