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
def iter_preceding(self) -> Iterator[Union[DocumentNode, ChildNodeType]]:
    """Iterator for 'preceding' reverse axis."""
    ancestors: Set[Union[ElementNode, DocumentNode]]
    item: XPathNode
    if isinstance(self.item, XPathNode):
        if self.document is not None or self.item is not self.root:
            item = self.item
            if (root := item.parent) is not None:
                status = (self.item, self.axis)
                self.axis = 'preceding'
                ancestors = {root}
                while root.parent is not None:
                    if root is self.root and self.document is None:
                        break
                    root = root.parent
                    ancestors.add(root)
                for self.item in root.iter_descendants():
                    if self.item is item:
                        break
                    if self.item not in ancestors:
                        yield self.item
                self.item, self.axis = status