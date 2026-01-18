from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
def parse_class_id(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
    """Parse HTML classes and ids."""
    selector = m.group(0)
    if selector.startswith('.'):
        sel.classes.append(css_unescape(selector[1:]))
    else:
        sel.ids.append(css_unescape(selector[1:]))
    has_selector = True
    return has_selector