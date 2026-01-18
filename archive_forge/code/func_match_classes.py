from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_classes(self, el: bs4.Tag, classes: tuple[str, ...]) -> bool:
    """Match element's classes."""
    current_classes = self.get_classes(el)
    found = True
    for c in classes:
        if c not in current_classes:
            found = False
            break
    return found