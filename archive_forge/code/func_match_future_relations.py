from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_future_relations(self, el: bs4.Tag, relation: ct.SelectorList) -> bool:
    """Match future relationship."""
    found = False
    if isinstance(relation[0], ct.SelectorNull):
        return found
    if relation[0].rel_type == REL_HAS_PARENT:
        found = self.match_future_child(el, relation, True)
    elif relation[0].rel_type == REL_HAS_CLOSE_PARENT:
        found = self.match_future_child(el, relation)
    elif relation[0].rel_type == REL_HAS_SIBLING:
        sibling = self.get_next(el)
        while not found and sibling:
            found = self.match_selectors(sibling, relation)
            sibling = self.get_next(sibling)
    elif relation[0].rel_type == REL_HAS_CLOSE_SIBLING:
        sibling = self.get_next(el)
        if sibling and self.is_tag(sibling):
            found = self.match_selectors(sibling, relation)
    return found