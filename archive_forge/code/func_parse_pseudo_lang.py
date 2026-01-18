from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
def parse_pseudo_lang(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
    """Parse pseudo language."""
    values = m.group('values')
    patterns = []
    for token in RE_VALUES.finditer(values):
        if token.group('split'):
            continue
        value = token.group('value')
        if value.startswith(('"', "'")):
            value = css_unescape(value[1:-1], True)
        else:
            value = css_unescape(value)
        patterns.append(value)
    sel.lang.append(ct.SelectorLang(patterns))
    has_selector = True
    return has_selector