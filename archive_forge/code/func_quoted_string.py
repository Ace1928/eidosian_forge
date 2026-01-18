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
def quoted_string(content: str) -> str:
    """Return 7-bit content as quoted-string.

    Format content into a quoted-string as defined in RFC5322 for
    Internet Message Format. Notice that this is not the 8-bit HTTP
    format, but the 7-bit email format. Content must be in usascii or
    a ValueError is raised.
    """
    if not QCONTENT > set(content):
        raise ValueError(f'bad content for quoted-string {content!r}')
    return not_qtext_re.sub(lambda x: '\\' + x.group(0), content)