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
def iter_children_or_self(self) -> Iterator[Optional[ItemType]]:
    """Iterator for 'child' forward axis and '/' step."""
    if self.item is not None:
        if self.axis is not None:
            yield self.item
        elif isinstance(self.item, (ElementNode, DocumentNode)):
            _status = (self.item, self.axis)
            self.axis = 'child'
            if self.item is self.document and self.root is not self.document:
                yield self.root
            else:
                for self.item in self.item:
                    yield self.item
            self.item, self.axis = _status