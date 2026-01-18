import calendar
import collections.abc
import copy
import datetime
import email.utils
from functools import lru_cache
from http.client import responses
import http.cookies
import re
from ssl import SSLError
import time
import unicodedata
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
from tornado.escape import native_str, parse_qs_bytes, utf8
from tornado.log import gen_log
from tornado.util import ObjectDict, unicode_type
import typing
from typing import (
def parse_body_arguments(content_type: str, body: bytes, arguments: Dict[str, List[bytes]], files: Dict[str, List[HTTPFile]], headers: Optional[HTTPHeaders]=None) -> None:
    """Parses a form request body.

    Supports ``application/x-www-form-urlencoded`` and
    ``multipart/form-data``.  The ``content_type`` parameter should be
    a string and ``body`` should be a byte string.  The ``arguments``
    and ``files`` parameters are dictionaries that will be updated
    with the parsed contents.
    """
    if content_type.startswith('application/x-www-form-urlencoded'):
        if headers and 'Content-Encoding' in headers:
            gen_log.warning('Unsupported Content-Encoding: %s', headers['Content-Encoding'])
            return
        try:
            uri_arguments = parse_qs_bytes(body, keep_blank_values=True)
        except Exception as e:
            gen_log.warning('Invalid x-www-form-urlencoded body: %s', e)
            uri_arguments = {}
        for name, values in uri_arguments.items():
            if values:
                arguments.setdefault(name, []).extend(values)
    elif content_type.startswith('multipart/form-data'):
        if headers and 'Content-Encoding' in headers:
            gen_log.warning('Unsupported Content-Encoding: %s', headers['Content-Encoding'])
            return
        try:
            fields = content_type.split(';')
            for field in fields:
                k, sep, v = field.strip().partition('=')
                if k == 'boundary' and v:
                    parse_multipart_form_data(utf8(v), body, arguments, files)
                    break
            else:
                raise ValueError('multipart boundary not found')
        except Exception as e:
            gen_log.warning('Invalid multipart/form-data: %s', e)