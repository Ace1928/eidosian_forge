import warnings
from contextlib import contextmanager
from typing import Optional, Tuple, Type
from redis import Connection as RedisConnection
from redis import Redis
from .local import LocalStack
def get_current_connection() -> 'Redis':
    """
    Returns the current Redis connection (i.e. the topmost on the
    connection stack).

    Returns:
        Redis: A Redis Connection
    """
    warnings.warn('The `get_current_connection` function is deprecated. Pass the `connection` explicitly instead.', DeprecationWarning)
    return _connection_stack.top