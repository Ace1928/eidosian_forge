from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def find_bidi(self, el: bs4.Tag) -> int | None:
    """Get directionality from element text."""
    for node in self.get_children(el, tags=False):
        if self.is_tag(node):
            direction = DIR_MAP.get(util.lower(self.get_attribute_by_name(node, 'dir', '')), None)
            if self.get_tag(node) in ('bdi', 'script', 'style', 'textarea', 'iframe') or not self.is_html_tag(node) or direction is not None:
                continue
            value = self.find_bidi(node)
            if value is not None:
                return value
            continue
        if self.is_special_string(node):
            continue
        for c in node:
            bidi = unicodedata.bidirectional(c)
            if bidi in ('AL', 'R', 'L'):
                return ct.SEL_DIR_LTR if bidi == 'L' else ct.SEL_DIR_RTL
    return None