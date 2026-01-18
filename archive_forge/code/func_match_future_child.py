from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_future_child(self, parent: bs4.Tag, relation: ct.SelectorList, recursive: bool=False) -> bool:
    """Match future child."""
    match = False
    if recursive:
        children = self.get_descendants
    else:
        children = self.get_children
    for child in children(parent, no_iframe=self.iframe_restrict):
        match = self.match_selectors(child, relation)
        if match:
            break
    return match