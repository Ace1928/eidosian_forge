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
def pubsub(self, **kwargs) -> 'PubSub':
    """
        Return a Publish/Subscribe object. With this object, you can
        subscribe to channels and listen for messages that get published to
        them.
        """
    return PubSub(self.connection_pool, **kwargs)