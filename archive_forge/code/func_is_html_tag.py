from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def is_html_tag(self, el: bs4.Tag) -> bool:
    """Check if tag is in HTML namespace."""
    return self.get_tag_ns(el) == NS_XHTML