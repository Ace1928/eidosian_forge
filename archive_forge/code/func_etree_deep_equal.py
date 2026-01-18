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
def etree_deep_equal(e1: ElementProtocol, e2: ElementProtocol) -> bool:
    if cm.ne(e1.tag, e2.tag):
        return False
    elif cm.ne((e1.text or '').strip(), (e2.text or '').strip()):
        return False
    elif cm.ne((e1.tail or '').strip(), (e2.tail or '').strip()):
        return False
    elif len(e1) != len(e2) or len(e1.attrib) != len(e2.attrib):
        return False
    try:
        items1 = {(cm.strxfrm(k or ''), cm.strxfrm(v)) for k, v in e1.attrib.items()}
        items2 = {(cm.strxfrm(k or ''), cm.strxfrm(v)) for k, v in e2.attrib.items()}
    except TypeError:
        return False
    if items1 != items2:
        return False
    return all((etree_deep_equal(c1, c2) for c1, c2 in zip(e1, e2)))