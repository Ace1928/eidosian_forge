from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_contains(self, el: bs4.Tag, contains: tuple[ct.SelectorContains, ...]) -> bool:
    """Match element if it contains text."""
    match = True
    content = None
    for contain_list in contains:
        if content is None:
            if contain_list.own:
                content = self.get_own_text(el, no_iframe=self.is_html)
            else:
                content = self.get_text(el, no_iframe=self.is_html)
        found = False
        for text in contain_list.text:
            if contain_list.own:
                for c in content:
                    if text in c:
                        found = True
                        break
                if found:
                    break
            elif text in content:
                found = True
                break
        if not found:
            match = False
    return match