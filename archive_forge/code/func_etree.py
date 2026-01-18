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
@property
def etree(self) -> ModuleType:
    if self._etree is None:
        if isinstance(self.root, (DocumentNode, ElementNode)):
            module_name = self.root.value.__class__.__module__
        elif isinstance(self.item, (DocumentNode, ElementNode, CommentNode, ProcessingInstructionNode)):
            module_name = self.item.value.__class__.__module__
        else:
            module_name = 'xml.etree.ElementTree'
        if not isinstance(module_name, str) or not module_name.startswith('lxml.'):
            etree_module_name = 'xml.etree.ElementTree'
        else:
            etree_module_name = 'lxml.etree'
        self._etree: ModuleType = importlib.import_module(etree_module_name)
    return self._etree