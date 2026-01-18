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
def get_context_item(self, item: ItemArgType, namespaces: Optional[NamespacesType]=None, uri: Optional[str]=None, fragment: bool=False) -> ItemType:
    """
        Checks the item and returns an item suitable for XPath processing.
        For XML trees and elements try a match with an existing node in the
        context. If it fails then builds a new node using also the provided
        optional arguments.
        """
    if isinstance(item, (XPathNode, AnyAtomicType)):
        return item
    elif is_etree_document(item):
        if self.root is not None and item is self.root.value:
            return self.root
        if self.documents:
            for doc in self.documents.values():
                if doc is not None and item is doc.value:
                    return doc
    elif is_etree_element(item):
        try:
            return self.root.elements[item]
        except (TypeError, KeyError, AttributeError):
            pass
        if self.documents:
            for doc in self.documents.values():
                if doc is not None and doc.elements is not None and (item in doc.elements):
                    return doc.elements[item]
        if callable(item.tag):
            if item.tag.__name__ == 'Comment':
                return CommentNode(cast(ElementProtocol, item))
            else:
                return ProcessingInstructionNode(cast(ElementProtocol, item))
    elif not isinstance(item, Token) or not callable(item):
        msg = f'Unexpected type {type(item)} for context item'
        raise ElementPathTypeError(msg)
    else:
        return item
    return get_node_tree(root=cast(Union[ElementProtocol, DocumentProtocol], item), namespaces=namespaces, uri=uri, fragment=fragment)