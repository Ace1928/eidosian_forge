import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def spublish(self, shard_channel: ChannelT, message: EncodableT) -> ResponseT:
    """
        Posts a message to the given shard channel.
        Returns the number of clients that received the message

        For more information see https://redis.io/commands/spublish
        """
    return self.execute_command('SPUBLISH', shard_channel, message)