from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def parse_grouping(p: str) -> tuple[int, int]:
    """Parse primary and secondary digit grouping

    >>> parse_grouping('##')
    (1000, 1000)
    >>> parse_grouping('#,###')
    (3, 3)
    >>> parse_grouping('#,####,###')
    (3, 4)
    """
    width = len(p)
    g1 = p.rfind(',')
    if g1 == -1:
        return (1000, 1000)
    g1 = width - g1 - 1
    g2 = p[:-g1 - 1].rfind(',')
    if g2 == -1:
        return (g1, g1)
    g2 = width - g1 - g2 - 2
    return (g1, g2)