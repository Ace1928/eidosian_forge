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
def register_connect_callback(self, callback):
    """
        Register a callback to be called when the connection is established either
        initially or reconnected.  This allows listeners to issue commands that
        are ephemeral to the connection, for example pub/sub subscription or
        key tracking.  The callback must be a _method_ and will be kept as
        a weak reference.
        """
    wm = weakref.WeakMethod(callback)
    if wm not in self._connect_callbacks:
        self._connect_callbacks.append(wm)