import time
import logging
import datetime
import functools
from pyzor.engines.common import *
def safe_call(f):
    """Decorator that wraps a method for handling database operations."""

    def wrapped_f(self, *args, **kwargs):
        try:
            return f(self, *args, **kwargs)
        except redis.exceptions.RedisError as e:
            self.log.error('Redis error while calling %s: %s', f.__name__, e)
            raise DatabaseError('Database temporarily unavailable.')
    return wrapped_f