from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_nth_tag_type(self, el: bs4.Tag, child: bs4.Tag) -> bool:
    """Match tag type for `nth` matches."""
    return self.get_tag(child) == self.get_tag(el) and self.get_tag_ns(child) == self.get_tag_ns(el)