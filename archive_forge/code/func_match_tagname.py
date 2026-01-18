from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_tagname(self, el: bs4.Tag, tag: ct.SelectorTag) -> bool:
    """Match tag name."""
    name = util.lower(tag.name) if not self.is_xml and tag.name is not None else tag.name
    return not (name is not None and name not in (self.get_tag(el), '*'))