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
@reify
def path_qs(self) -> str:
    """The URL including PATH_INFO and the query string.

        E.g, /app/blog?id=10
        """
    return str(self._rel_url)