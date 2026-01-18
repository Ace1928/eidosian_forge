import decimal
import math
from copy import copy
from decimal import Decimal
from itertools import product
from typing import TYPE_CHECKING, cast, Dict, Optional, List, Tuple, \
import urllib.parse
from .exceptions import ElementPathError, ElementPathValueError, \
from .helpers import ordinal, get_double, split_function_test
from .etree import is_etree_element, is_etree_document
from .namespaces import XSD_NAMESPACE, XPATH_FUNCTIONS_NAMESPACE, \
from .tree_builders import get_node_tree
from .xpath_nodes import XPathNode, ElementNode, AttributeNode, \
from .datatypes import xsd10_atomic_types, AbstractDateTime, AnyURI, \
from .protocols import ElementProtocol, DocumentProtocol, XsdAttributeProtocol, \
from .sequence_types import is_sequence_type_restriction, match_sequence_type
from .schema_proxy import AbstractSchemaProxy
from .tdop import Token, MultiLabel
from .xpath_context import XPathContext, XPathSchemaContext
def iter_leaf_elements(self) -> Iterator[str]:
    """
        Iterates through the leaf elements of the token tree if there are any,
        returning QNames in prefixed format. A leaf element is an element
        positioned at last path step. Does not consider kind tests and wildcards.
        """
    if self.symbol in ('(name)', ':'):
        yield cast(str, self.value)
    elif self.symbol in ('//', '/'):
        if self._items[-1].symbol in _LEAF_ELEMENTS_TOKENS:
            yield from self._items[-1].iter_leaf_elements()
    elif self.symbol in ('[',):
        yield from self._items[0].iter_leaf_elements()
    else:
        for tk in self._items:
            yield from tk.iter_leaf_elements()