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
def is_empty_sequence(x: Any) -> bool:
    return not x and isinstance(x, list)