from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_namespace(self, el: bs4.Tag, tag: ct.SelectorTag) -> bool:
    """Match the namespace of the element."""
    match = True
    namespace = self.get_tag_ns(el)
    default_namespace = self.namespaces.get('')
    tag_ns = '' if tag.prefix is None else self.namespaces.get(tag.prefix)
    if tag.prefix is None and (default_namespace is not None and namespace != default_namespace):
        match = False
    elif tag.prefix is not None and tag.prefix == '' and namespace:
        match = False
    elif tag.prefix and tag.prefix != '*' and (tag_ns is None or namespace != tag_ns):
        match = False
    return match