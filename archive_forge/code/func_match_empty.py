from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_empty(self, el: bs4.Tag) -> bool:
    """Check if element is empty (if requested)."""
    is_empty = True
    for child in self.get_children(el, tags=False):
        if self.is_tag(child):
            is_empty = False
            break
        elif self.is_content_string(child) and RE_NOT_EMPTY.search(child):
            is_empty = False
            break
    return is_empty