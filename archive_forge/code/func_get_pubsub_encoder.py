import uuid
import logging
import asyncio
import copy
import enum
import errno
import inspect
import io
import os
import socket
import ssl
import threading
import weakref
from itertools import chain
from types import MappingProxyType
from typing import (
from urllib.parse import ParseResult, parse_qs, unquote, urlparse
import async_timeout
from aiokeydb.v1.backoff import NoBackoff
from aiokeydb.v1.asyncio.retry import Retry
from aiokeydb.v1.compat import Protocol, TypedDict
from aiokeydb.v1.exceptions import (
from aiokeydb.v1.credentials import CredentialProvider, UsernamePasswordCredentialProvider
from aiokeydb.v1.typing import EncodableT, EncodedT
from aiokeydb.v1.utils import HIREDIS_AVAILABLE, str_if_bytes, set_ulimits
def get_pubsub_encoder(self):
    """
        Returns the pubsub encoder based on encoding settings
        """
    kwargs = self.connection_kwargs
    return self.encoder_class(encoding=kwargs.get('encoding', 'utf-8'), encoding_errors=kwargs.get('encoding_errors', 'strict'), decode_responses=self.pubsub_decode_responses or kwargs.get('decode_responses', True))