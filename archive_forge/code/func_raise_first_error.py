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
def raise_first_error(self, commands: CommandStackT, response: Iterable[Any]):
    for i, r in enumerate(response):
        if isinstance(r, ResponseError):
            self.annotate_exception(r, i + 1, commands[i][0])
            raise r