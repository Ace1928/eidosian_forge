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
def rfc822_formatted_time() -> str:
    global _cached_current_datetime
    global _cached_formatted_datetime
    now = int(time.time())
    if now != _cached_current_datetime:
        _weekdayname = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')
        _monthname = ('', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
        year, month, day, hh, mm, ss, wd, *tail = time.gmtime(now)
        _cached_formatted_datetime = '%s, %02d %3s %4d %02d:%02d:%02d GMT' % (_weekdayname[wd], day, _monthname[month], year, hh, mm, ss)
        _cached_current_datetime = now
    return _cached_formatted_datetime