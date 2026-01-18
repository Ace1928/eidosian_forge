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
def raw_path(self) -> str:
    """The URL including raw *PATH INFO* without the host or scheme.

        Warning, the path is unquoted and may contains non valid URL characters

        E.g., ``/my%2Fpath%7Cwith%21some%25strange%24characters``
        """
    return self._message.path