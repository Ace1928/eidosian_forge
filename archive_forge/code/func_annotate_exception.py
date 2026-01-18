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
def annotate_exception(self, exception: Exception, number: int, command: Iterable[object]) -> None:
    cmd = ' '.join(map(safe_str, command))
    msg = f'Command # {number} ({cmd}) of pipeline caused error: {exception.args}'
    exception.args = (msg,) + exception.args[1:]