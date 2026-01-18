from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
def parse_attribute_selector(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
    """Create attribute selector from the returned regex match."""
    inverse = False
    op = m.group('cmp')
    case = util.lower(m.group('case')) if m.group('case') else None
    ns = css_unescape(m.group('attr_ns')[:-1]) if m.group('attr_ns') else ''
    attr = css_unescape(m.group('attr_name'))
    is_type = False
    pattern2 = None
    value = ''
    if case:
        flags = (re.I if case == 'i' else 0) | re.DOTALL
    elif util.lower(attr) == 'type':
        flags = re.I | re.DOTALL
        is_type = True
    else:
        flags = re.DOTALL
    if op:
        if m.group('value').startswith(('"', "'")):
            value = css_unescape(m.group('value')[1:-1], True)
        else:
            value = css_unescape(m.group('value'))
    if not op:
        pattern = None
    elif op.startswith('^'):
        pattern = re.compile('^%s.*' % re.escape(value), flags)
    elif op.startswith('$'):
        pattern = re.compile('.*?%s$' % re.escape(value), flags)
    elif op.startswith('*'):
        pattern = re.compile('.*?%s.*' % re.escape(value), flags)
    elif op.startswith('~'):
        value = '[^\\s\\S]' if not value or RE_WS.search(value) else re.escape(value)
        pattern = re.compile('.*?(?:(?<=^)|(?<=[ \\t\\r\\n\\f]))%s(?=(?:[ \\t\\r\\n\\f]|$)).*' % value, flags)
    elif op.startswith('|'):
        pattern = re.compile('^%s(?:-.*)?$' % re.escape(value), flags)
    else:
        pattern = re.compile('^%s$' % re.escape(value), flags)
        if op.startswith('!'):
            inverse = True
    if is_type and pattern:
        pattern2 = re.compile(pattern.pattern)
    sel_attr = ct.SelectorAttribute(attr, ns, pattern, pattern2)
    if inverse:
        sub_sel = _Selector()
        sub_sel.attributes.append(sel_attr)
        not_list = ct.SelectorList([sub_sel.freeze()], True, False)
        sel.selectors.append(not_list)
    else:
        sel.attributes.append(sel_attr)
    has_selector = True
    return has_selector