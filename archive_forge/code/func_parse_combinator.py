from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
def parse_combinator(self, sel: _Selector, m: Match[str], has_selector: bool, selectors: list[_Selector], relations: list[_Selector], is_pseudo: bool, is_forgive: bool, index: int) -> tuple[bool, _Selector]:
    """Parse combinator tokens."""
    combinator = m.group('relation').strip()
    if not combinator:
        combinator = WS_COMBINATOR
    if not has_selector:
        if not is_forgive or combinator != COMMA_COMBINATOR:
            raise SelectorSyntaxError(f"The combinator '{combinator}' at position {index}, must have a selector before it", self.pattern, index)
        if combinator == COMMA_COMBINATOR:
            sel.no_match = True
            del relations[:]
            selectors.append(sel)
    elif combinator == COMMA_COMBINATOR:
        if not sel.tag and (not is_pseudo):
            sel.tag = ct.SelectorTag('*', None)
        sel.relations.extend(relations)
        selectors.append(sel)
        del relations[:]
    else:
        sel.relations.extend(relations)
        sel.rel_type = combinator
        del relations[:]
        relations.append(sel)
    sel = _Selector()
    has_selector = False
    return (has_selector, sel)