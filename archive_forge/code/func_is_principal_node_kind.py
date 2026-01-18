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
def is_principal_node_kind(self) -> bool:
    if self.axis == 'attribute':
        return isinstance(self.item, AttributeNode)
    elif self.axis == 'namespace':
        return isinstance(self.item, NamespaceNode)
    else:
        return isinstance(self.item, ElementNode)