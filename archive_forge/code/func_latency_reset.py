import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def latency_reset(self, *events: str) -> ResponseT:
    """
        Resets the latency spikes time series of all, or only some, events.

        For more information see https://redis.io/commands/latency-reset
        """
    return self.execute_command('LATENCY RESET', *events)