from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def parse_precision(p):
    """Calculate the min and max allowed digits"""
    min = max = 0
    for c in p:
        if c in '@0':
            min += 1
            max += 1
        elif c == '#':
            max += 1
        elif c == ',':
            continue
        else:
            break
    return (min, max)