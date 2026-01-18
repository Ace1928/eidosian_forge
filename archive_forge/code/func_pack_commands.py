import asyncio
import copy
import enum
import inspect
import socket
import ssl
import sys
import warnings
import weakref
from abc import abstractmethod
from itertools import chain
from types import MappingProxyType
from typing import (
from urllib.parse import ParseResult, parse_qs, unquote, urlparse
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.compat import Protocol, TypedDict
from redis.connection import DEFAULT_RESP_VERSION
from redis.credentials import CredentialProvider, UsernamePasswordCredentialProvider
from redis.exceptions import (
from redis.typing import EncodableT
from redis.utils import HIREDIS_AVAILABLE, get_lib_version, str_if_bytes
from .._parsers import (
def pack_commands(self, commands: Iterable[Iterable[EncodableT]]) -> List[bytes]:
    """Pack multiple commands into the Redis protocol"""
    output: List[bytes] = []
    pieces: List[bytes] = []
    buffer_length = 0
    buffer_cutoff = self._buffer_cutoff
    for cmd in commands:
        for chunk in self.pack_command(*cmd):
            chunklen = len(chunk)
            if buffer_length > buffer_cutoff or chunklen > buffer_cutoff or isinstance(chunk, memoryview):
                if pieces:
                    output.append(SYM_EMPTY.join(pieces))
                buffer_length = 0
                pieces = []
            if chunklen > buffer_cutoff or isinstance(chunk, memoryview):
                output.append(chunk)
            else:
                pieces.append(chunk)
                buffer_length += chunklen
    if pieces:
        output.append(SYM_EMPTY.join(pieces))
    return output