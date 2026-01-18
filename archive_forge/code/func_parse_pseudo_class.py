from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
def parse_pseudo_class(self, sel: _Selector, m: Match[str], has_selector: bool, iselector: Iterator[tuple[str, Match[str]]], is_html: bool) -> tuple[bool, bool]:
    """Parse pseudo class."""
    complex_pseudo = False
    pseudo = util.lower(css_unescape(m.group('name')))
    if m.group('open'):
        complex_pseudo = True
    if complex_pseudo and pseudo in PSEUDO_COMPLEX:
        has_selector = self.parse_pseudo_open(sel, pseudo, has_selector, iselector, m.end(0))
    elif not complex_pseudo and pseudo in PSEUDO_SIMPLE:
        if pseudo == ':root':
            sel.flags |= ct.SEL_ROOT
        elif pseudo == ':defined':
            sel.flags |= ct.SEL_DEFINED
            is_html = True
        elif pseudo == ':scope':
            sel.flags |= ct.SEL_SCOPE
        elif pseudo == ':empty':
            sel.flags |= ct.SEL_EMPTY
        elif pseudo in (':link', ':any-link'):
            sel.selectors.append(CSS_LINK)
        elif pseudo == ':checked':
            sel.selectors.append(CSS_CHECKED)
        elif pseudo == ':default':
            sel.selectors.append(CSS_DEFAULT)
        elif pseudo == ':indeterminate':
            sel.selectors.append(CSS_INDETERMINATE)
        elif pseudo == ':disabled':
            sel.selectors.append(CSS_DISABLED)
        elif pseudo == ':enabled':
            sel.selectors.append(CSS_ENABLED)
        elif pseudo == ':required':
            sel.selectors.append(CSS_REQUIRED)
        elif pseudo == ':optional':
            sel.selectors.append(CSS_OPTIONAL)
        elif pseudo == ':read-only':
            sel.selectors.append(CSS_READ_ONLY)
        elif pseudo == ':read-write':
            sel.selectors.append(CSS_READ_WRITE)
        elif pseudo == ':in-range':
            sel.selectors.append(CSS_IN_RANGE)
        elif pseudo == ':out-of-range':
            sel.selectors.append(CSS_OUT_OF_RANGE)
        elif pseudo == ':placeholder-shown':
            sel.selectors.append(CSS_PLACEHOLDER_SHOWN)
        elif pseudo == ':first-child':
            sel.nth.append(ct.SelectorNth(1, False, 0, False, False, ct.SelectorList()))
        elif pseudo == ':last-child':
            sel.nth.append(ct.SelectorNth(1, False, 0, False, True, ct.SelectorList()))
        elif pseudo == ':first-of-type':
            sel.nth.append(ct.SelectorNth(1, False, 0, True, False, ct.SelectorList()))
        elif pseudo == ':last-of-type':
            sel.nth.append(ct.SelectorNth(1, False, 0, True, True, ct.SelectorList()))
        elif pseudo == ':only-child':
            sel.nth.extend([ct.SelectorNth(1, False, 0, False, False, ct.SelectorList()), ct.SelectorNth(1, False, 0, False, True, ct.SelectorList())])
        elif pseudo == ':only-of-type':
            sel.nth.extend([ct.SelectorNth(1, False, 0, True, False, ct.SelectorList()), ct.SelectorNth(1, False, 0, True, True, ct.SelectorList())])
        has_selector = True
    elif complex_pseudo and pseudo in PSEUDO_COMPLEX_NO_MATCH:
        self.parse_selectors(iselector, m.end(0), FLG_PSEUDO | FLG_OPEN)
        sel.no_match = True
        has_selector = True
    elif not complex_pseudo and pseudo in PSEUDO_SIMPLE_NO_MATCH:
        sel.no_match = True
        has_selector = True
    elif pseudo in PSEUDO_SUPPORTED:
        raise SelectorSyntaxError(f"Invalid syntax for pseudo class '{pseudo}'", self.pattern, m.start(0))
    else:
        raise NotImplementedError(f"'{pseudo}' pseudo-class is not implemented at this time")
    return (has_selector, is_html)