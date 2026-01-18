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
def split_interval_pattern(pattern: str) -> list[str]:
    """
    Split an interval-describing datetime pattern into multiple pieces.

    > The pattern is then designed to be broken up into two pieces by determining the first repeating field.
    - https://www.unicode.org/reports/tr35/tr35-dates.html#intervalFormats

    >>> split_interval_pattern(u'E d.M. – E d.M.')
    [u'E d.M. – ', 'E d.M.']
    >>> split_interval_pattern("Y 'text' Y 'more text'")
    ["Y 'text '", "Y 'more text'"]
    >>> split_interval_pattern(u"E, MMM d – E")
    [u'E, MMM d – ', u'E']
    >>> split_interval_pattern("MMM d")
    ['MMM d']
    >>> split_interval_pattern("y G")
    ['y G']
    >>> split_interval_pattern(u"MMM d – d")
    [u'MMM d – ', u'd']

    :param pattern: Interval pattern string
    :return: list of "subpatterns"
    """
    seen_fields = set()
    parts = [[]]
    for tok_type, tok_value in tokenize_pattern(pattern):
        if tok_type == 'field':
            if tok_value[0] in seen_fields:
                parts.append([])
                seen_fields.clear()
            seen_fields.add(tok_value[0])
        parts[-1].append((tok_type, tok_value))
    return [untokenize_pattern(tokens) for tokens in parts]