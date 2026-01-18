import hashlib
import inspect
import logging
from contextlib import contextmanager, asynccontextmanager
from typing import Union, Optional, List, Callable, Generator, AsyncGenerator, TYPE_CHECKING
from aiokeydb.v2.types import ENOVAL
from redis.utils import (
def set_ulimits(max_connections: int=500, verbose: bool=False):
    """
    Sets the system ulimits
    to allow for the maximum number of open connections

    - if the current ulimit > max_connections, then it is ignored
    - if it is less, then we set it.
    """
    import resource
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit > max_connections:
        return
    if hard_limit < max_connections and verbose:
        logger.warning(f'The current hard limit ({hard_limit}) is less than max_connections ({max_connections}).')
    new_hard_limit = max(hard_limit, max_connections)
    if verbose:
        logger.info(f'Setting new ulimits to ({soft_limit}, {hard_limit}) -> ({max_connections}, {new_hard_limit})')
    resource.setrlimit(resource.RLIMIT_NOFILE, (max_connections + 10, new_hard_limit))
    new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if verbose:
        logger.info(f'New Limits: ({new_soft}, {new_hard})')