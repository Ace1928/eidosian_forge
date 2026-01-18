import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def pubsub_numsub(self, *args: ChannelT, **kwargs) -> ResponseT:
    """
        Return a list of (channel, number of subscribers) tuples
        for each channel given in ``*args``

        For more information see https://redis.io/commands/pubsub-numsub
        """
    return self.execute_command('PUBSUB NUMSUB', *args, **kwargs)