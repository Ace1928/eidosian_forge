import warnings
from contextlib import contextmanager
from typing import Optional, Tuple, Type
from redis import Connection as RedisConnection
from redis import Redis
from .local import LocalStack
def push_connection(redis: 'Redis'):
    """
    Pushes the given connection to the stack.

    Args:
        redis (Redis): A Redis connection
    """
    warnings.warn('The `push_connection` function is deprecated. Pass the `connection` explicitly instead.', DeprecationWarning)
    _connection_stack.push(redis)