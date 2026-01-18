import asyncio
import copy
import inspect
import re
import ssl
import warnings
from typing import (
from redis._parsers.helpers import (
from redis.asyncio.connection import (
from redis.asyncio.lock import Lock
from redis.asyncio.retry import Retry
from redis.client import (
from redis.commands import (
from redis.compat import Protocol, TypedDict
from redis.credentials import CredentialProvider
from redis.exceptions import (
from redis.typing import ChannelT, EncodableT, KeyT
from redis.utils import (
def multi(self):
    """
        Start a transactional block of the pipeline after WATCH commands
        are issued. End the transactional block with `execute`.
        """
    if self.explicit_transaction:
        raise RedisError('Cannot issue nested calls to MULTI')
    if self.command_stack:
        raise RedisError('Commands without an initial WATCH have already been issued')
    self.explicit_transaction = True