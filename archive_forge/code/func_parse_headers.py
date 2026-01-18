import abc
import asyncio
import re
import string
from contextlib import suppress
from enum import IntEnum
from typing import (
from multidict import CIMultiDict, CIMultiDictProxy, istr
from yarl import URL
from . import hdrs
from .base_protocol import BaseProtocol
from .compression_utils import HAS_BROTLI, BrotliDecompressor, ZLibDecompressor
from .helpers import (
from .http_exceptions import (
from .http_writer import HttpVersion, HttpVersion10
from .log import internal_logger
from .streams import EMPTY_PAYLOAD, StreamReader
from .typedefs import RawHeaders
def parse_headers(self, lines: List[bytes]) -> Tuple['CIMultiDictProxy[str]', RawHeaders, Optional[bool], Optional[str], bool, bool]:
    """Parses RFC 5322 headers from a stream.

        Line continuations are supported. Returns list of header name
        and value pairs. Header name is in upper case.
        """
    headers, raw_headers = self._headers_parser.parse_headers(lines)
    close_conn = None
    encoding = None
    upgrade = False
    chunked = False
    singletons = (hdrs.CONTENT_LENGTH, hdrs.CONTENT_LOCATION, hdrs.CONTENT_RANGE, hdrs.CONTENT_TYPE, hdrs.ETAG, hdrs.HOST, hdrs.MAX_FORWARDS, hdrs.SERVER, hdrs.TRANSFER_ENCODING, hdrs.USER_AGENT)
    bad_hdr = next((h for h in singletons if len(headers.getall(h, ())) > 1), None)
    if bad_hdr is not None:
        raise BadHttpMessage(f"Duplicate '{bad_hdr}' header found.")
    conn = headers.get(hdrs.CONNECTION)
    if conn:
        v = conn.lower()
        if v == 'close':
            close_conn = True
        elif v == 'keep-alive':
            close_conn = False
        elif v == 'upgrade' and headers.get(hdrs.UPGRADE):
            upgrade = True
    enc = headers.get(hdrs.CONTENT_ENCODING)
    if enc:
        enc = enc.lower()
        if enc in ('gzip', 'deflate', 'br'):
            encoding = enc
    te = headers.get(hdrs.TRANSFER_ENCODING)
    if te is not None:
        if 'chunked' == te.lower():
            chunked = True
        else:
            raise BadHttpMessage('Request has invalid `Transfer-Encoding`')
        if hdrs.CONTENT_LENGTH in headers:
            raise BadHttpMessage("Transfer-Encoding can't be present with Content-Length")
    return (headers, raw_headers, close_conn, encoding, upgrade, chunked)