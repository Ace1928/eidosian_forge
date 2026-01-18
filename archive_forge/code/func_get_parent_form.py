from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def get_parent_form(el: bs4.Tag) -> bs4.Tag | None:
    """Find this input's form."""
    form = None
    parent = self.get_parent(el, no_iframe=True)
    while form is None:
        if self.get_tag(parent) == 'form' and self.is_html_tag(parent):
            form = parent
            break
        last_parent = parent
        parent = self.get_parent(parent, no_iframe=True)
        if parent is None:
            form = last_parent
            break
    return form