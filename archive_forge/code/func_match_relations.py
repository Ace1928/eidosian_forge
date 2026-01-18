from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_relations(self, el: bs4.Tag, relation: ct.SelectorList) -> bool:
    """Match relationship to other elements."""
    found = False
    if isinstance(relation[0], ct.SelectorNull) or relation[0].rel_type is None:
        return found
    if relation[0].rel_type.startswith(':'):
        found = self.match_future_relations(el, relation)
    else:
        found = self.match_past_relations(el, relation)
    return found