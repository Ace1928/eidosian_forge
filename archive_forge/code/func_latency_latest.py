import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def latency_latest(self) -> ResponseT:
    """
        Reports the latest latency events logged.

        For more information see https://redis.io/commands/latency-latest
        """
    return self.execute_command('LATENCY LATEST')