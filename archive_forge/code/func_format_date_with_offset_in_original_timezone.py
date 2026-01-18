import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def format_date_with_offset_in_original_timezone(t, offset=0, _cache=_offset_cache):
    """Return a formatted date string in the original timezone.

    This routine may be faster then format_date.

    :param t: Seconds since the epoch.
    :param offset: Timezone offset in seconds east of utc.
    """
    if offset is None:
        offset = 0
    tt = time.gmtime(t + offset)
    date_fmt = _default_format_by_weekday_num[tt[6]]
    date_str = time.strftime(date_fmt, tt)
    offset_str = _cache.get(offset, None)
    if offset_str is None:
        offset_str = ' %+03d%02d' % (offset / 3600, offset / 60 % 60)
        _cache[offset] = offset_str
    return date_str + offset_str