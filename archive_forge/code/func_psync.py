import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def psync(self, replicationid: str, offset: int):
    """
        Initiates a replication stream from the master.
        Newer version for `sync`.

        For more information see https://redis.io/commands/sync
        """
    from redis.client import NEVER_DECODE
    options = {}
    options[NEVER_DECODE] = []
    return self.execute_command('PSYNC', replicationid, offset, **options)