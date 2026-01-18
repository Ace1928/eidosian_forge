import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def latency_history(self, event: str) -> ResponseT:
    """
        Returns the raw data of the ``event``'s latency spikes time series.

        For more information see https://redis.io/commands/latency-history
        """
    return self.execute_command('LATENCY HISTORY', event)