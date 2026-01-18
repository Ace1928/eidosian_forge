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
def compare_func(obj1: Any, obj2: Any) -> int:
    if key_func is not None:
        if isinstance(obj1, (list, Iterator)):
            obj1 = map(key_func, obj1)
        else:
            obj1 = key_func(obj1)
        if isinstance(obj2, (list, Iterator)):
            obj2 = map(key_func, obj2)
        else:
            obj2 = key_func(obj2)
    return deep_compare(obj1, obj2, collation, token)