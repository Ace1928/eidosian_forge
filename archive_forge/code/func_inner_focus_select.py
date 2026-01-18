import datetime
import importlib
from copy import copy
from types import ModuleType
from typing import TYPE_CHECKING, cast, Dict, Any, List, Iterator, \
from .exceptions import ElementPathTypeError
from .tdop import Token
from .namespaces import NamespacesType
from .datatypes import AnyAtomicType, Timezone, Language
from .protocols import ElementProtocol, DocumentProtocol
from .etree import is_etree_element, is_etree_document
from .xpath_nodes import ChildNodeType, XPathNode, AttributeNode, NamespaceNode, \
from .tree_builders import RootArgType, get_node_tree
def inner_focus_select(self, token: Union['XPathToken', 'XPathAxis']) -> Iterator[Any]:
    """Apply the token's selector with an inner focus."""
    status = (self.item, self.size, self.position, self.axis)
    results = [x for x in token.select(copy(self))]
    self.axis = None
    if token.label == 'axis' and cast('XPathAxis', token).reverse_axis:
        self.size = self.position = len(results)
        for self.item in results:
            yield self.item
            self.position -= 1
    else:
        self.size = len(results)
        for self.position, self.item in enumerate(results, start=1):
            yield self.item
    self.item, self.size, self.position, self.axis = status