import math
from decimal import Decimal
from functools import cmp_to_key
from itertools import zip_longest
from typing import Any, Callable, Optional, Iterable, Iterator
from .protocols import ElementProtocol
from .exceptions import xpath_error
from .datatypes import UntypedAtomic, AnyURI, AbstractQName
from .collations import UNICODE_CODEPOINT_COLLATION, CollationManager
from .xpath_nodes import XPathNode, ElementNode, AttributeNode, NamespaceNode, \
from .xpath_tokens import XPathToken, XPathFunction, XPathMap, XPathArray
def etree_deep_compare(e1: ElementProtocol, e2: ElementProtocol) -> int:
    nonlocal result
    result = cm.strcoll(e1.tag, e2.tag)
    if result:
        return result
    result = cm.strcoll((e1.text or '').strip(), (e2.text or '').strip())
    if result:
        return result
    for a1, a2 in zip_longest(e1.attrib.items(), e2.attrib.items()):
        if a1 is None:
            return 1
        elif a2 is None:
            return -1
        result = cm.strcoll(a1[0], a2[0]) or cm.strcoll(a1[1], a2[1])
        if result:
            return result
    for c1, c2 in zip_longest(e1, e2):
        if c1 is None:
            return 1
        elif c2 is None:
            return -1
        result = etree_deep_compare(c1, c2)
        if result:
            return result
    else:
        result = cm.strcoll((e1.tail or '').strip(), (e2.tail or '').strip())
        if result:
            return result
        return 0