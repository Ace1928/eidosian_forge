import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def latency_histogram(self, *args):
    """
        This function throws a NotImplementedError since it is intentionally
        not supported.
        """
    raise NotImplementedError('LATENCY HISTOGRAM is intentionally not implemented in the client.')