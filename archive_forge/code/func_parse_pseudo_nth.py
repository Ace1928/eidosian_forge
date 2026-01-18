from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
def parse_pseudo_nth(self, sel: _Selector, m: Match[str], has_selector: bool, iselector: Iterator[tuple[str, Match[str]]]) -> bool:
    """Parse `nth` pseudo."""
    mdict = m.groupdict()
    if mdict.get('pseudo_nth_child'):
        postfix = '_child'
    else:
        postfix = '_type'
    mdict['name'] = util.lower(css_unescape(mdict['name']))
    content = util.lower(mdict.get('nth' + postfix))
    if content == 'even':
        s1 = 2
        s2 = 0
        var = True
    elif content == 'odd':
        s1 = 2
        s2 = 1
        var = True
    else:
        nth_parts = cast(Match[str], RE_NTH.match(content))
        _s1 = '-' if nth_parts.group('s1') and nth_parts.group('s1') == '-' else ''
        a = nth_parts.group('a')
        var = a.endswith('n')
        if a.startswith('n'):
            _s1 += '1'
        elif var:
            _s1 += a[:-1]
        else:
            _s1 += a
        _s2 = '-' if nth_parts.group('s2') and nth_parts.group('s2') == '-' else ''
        if nth_parts.group('b'):
            _s2 += nth_parts.group('b')
        else:
            _s2 = '0'
        s1 = int(_s1, 10)
        s2 = int(_s2, 10)
    pseudo_sel = mdict['name']
    if postfix == '_child':
        if m.group('of'):
            nth_sel = self.parse_selectors(iselector, m.end(0), FLG_PSEUDO | FLG_OPEN)
        else:
            nth_sel = CSS_NTH_OF_S_DEFAULT
        if pseudo_sel == ':nth-child':
            sel.nth.append(ct.SelectorNth(s1, var, s2, False, False, nth_sel))
        elif pseudo_sel == ':nth-last-child':
            sel.nth.append(ct.SelectorNth(s1, var, s2, False, True, nth_sel))
    elif pseudo_sel == ':nth-of-type':
        sel.nth.append(ct.SelectorNth(s1, var, s2, True, False, ct.SelectorList()))
    elif pseudo_sel == ':nth-last-of-type':
        sel.nth.append(ct.SelectorNth(s1, var, s2, True, True, ct.SelectorList()))
    has_selector = True
    return has_selector