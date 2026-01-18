from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_id(self, el: bs4.Tag, ids: tuple[str, ...]) -> bool:
    """Match element's ID."""
    found = True
    for i in ids:
        if i != self.get_attribute_by_name(el, 'id', ''):
            found = False
            break
    return found