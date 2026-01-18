import asyncio
import datetime
import io
import re
import socket
import string
import tempfile
import types
import warnings
from http.cookies import SimpleCookie
from types import MappingProxyType
from typing import (
from urllib.parse import parse_qsl
import attr
from multidict import (
from yarl import URL
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import (
from .http_parser import RawRequestMessage
from .http_writer import HttpVersion
from .multipart import BodyPartReader, MultipartReader
from .streams import EmptyStreamReader, StreamReader
from .typedefs import (
from .web_exceptions import HTTPRequestEntityTooLarge
from .web_response import StreamResponse
@property
def can_read_body(self) -> bool:
    """Return True if request's HTTP BODY can be read, False otherwise."""
    return not self._payload.at_eof()