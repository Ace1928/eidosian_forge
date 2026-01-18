from __future__ import annotations
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, SupportsInt
import datetime
from collections.abc import Iterable
from babel import localtime
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def append_chars():
    result.append(('chars', ''.join(charbuf).replace('\x00', "'")))
    del charbuf[:]