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
def forwarded(self) -> Tuple[Mapping[str, str], ...]:
    """A tuple containing all parsed Forwarded header(s).

        Makes an effort to parse Forwarded headers as specified by RFC 7239:

        - It adds one (immutable) dictionary per Forwarded 'field-value', ie
          per proxy. The element corresponds to the data in the Forwarded
          field-value added by the first proxy encountered by the client. Each
          subsequent item corresponds to those added by later proxies.
        - It checks that every value has valid syntax in general as specified
          in section 4: either a 'token' or a 'quoted-string'.
        - It un-escapes found escape sequences.
        - It does NOT validate 'by' and 'for' contents as specified in section
          6.
        - It does NOT validate 'host' contents (Host ABNF).
        - It does NOT validate 'proto' contents for valid URI scheme names.

        Returns a tuple containing one or more immutable dicts
        """
    elems = []
    for field_value in self._message.headers.getall(hdrs.FORWARDED, ()):
        length = len(field_value)
        pos = 0
        need_separator = False
        elem: Dict[str, str] = {}
        elems.append(types.MappingProxyType(elem))
        while 0 <= pos < length:
            match = _FORWARDED_PAIR_RE.match(field_value, pos)
            if match is not None:
                if need_separator:
                    pos = field_value.find(',', pos)
                else:
                    name, value, port = match.groups()
                    if value[0] == '"':
                        value = _QUOTED_PAIR_REPLACE_RE.sub('\\1', value[1:-1])
                    if port:
                        value += port
                    elem[name.lower()] = value
                    pos += len(match.group(0))
                    need_separator = True
            elif field_value[pos] == ',':
                need_separator = False
                elem = {}
                elems.append(types.MappingProxyType(elem))
                pos += 1
            elif field_value[pos] == ';':
                need_separator = False
                pos += 1
            elif field_value[pos] in ' \t':
                pos += 1
            else:
                pos = field_value.find(',', pos)
    return tuple(elems)