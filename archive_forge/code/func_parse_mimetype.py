import asyncio
import base64
import binascii
import contextlib
import datetime
import enum
import functools
import inspect
import netrc
import os
import platform
import re
import sys
import time
import warnings
import weakref
from collections import namedtuple
from contextlib import suppress
from email.parser import HeaderParser
from email.utils import parsedate
from math import ceil
from pathlib import Path
from types import TracebackType
from typing import (
from urllib.parse import quote
from urllib.request import getproxies, proxy_bypass
import attr
from multidict import MultiDict, MultiDictProxy, MultiMapping
from yarl import URL
from . import hdrs
from .log import client_logger, internal_logger
@functools.lru_cache(maxsize=56)
def parse_mimetype(mimetype: str) -> MimeType:
    """Parses a MIME type into its components.

    mimetype is a MIME type string.

    Returns a MimeType object.

    Example:

    >>> parse_mimetype('text/html; charset=utf-8')
    MimeType(type='text', subtype='html', suffix='',
             parameters={'charset': 'utf-8'})

    """
    if not mimetype:
        return MimeType(type='', subtype='', suffix='', parameters=MultiDictProxy(MultiDict()))
    parts = mimetype.split(';')
    params: MultiDict[str] = MultiDict()
    for item in parts[1:]:
        if not item:
            continue
        key, _, value = item.partition('=')
        params.add(key.lower().strip(), value.strip(' "'))
    fulltype = parts[0].strip().lower()
    if fulltype == '*':
        fulltype = '*/*'
    mtype, _, stype = fulltype.partition('/')
    stype, _, suffix = stype.partition('+')
    return MimeType(type=mtype, subtype=stype, suffix=suffix, parameters=MultiDictProxy(params))