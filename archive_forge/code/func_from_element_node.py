from importlib import import_module
from urllib.parse import urljoin
from types import ModuleType
from typing import cast, Any, Dict, Iterator, List, MutableMapping, Optional, Tuple, Union
from .datatypes import UntypedAtomic, get_atomic_value, AtomicValueType
from .namespaces import XML_NAMESPACE, XML_BASE, XSI_NIL, \
from .protocols import ElementProtocol, DocumentProtocol, XsdElementProtocol, \
from .helpers import match_wildcard, is_absolute_uri
from .etree import etree_iter_strings, is_etree_element, is_etree_document
@classmethod
def from_element_node(cls, root_node: ElementNode, replace: bool=True) -> 'DocumentNode':
    """
        Build a `DocumentNode` from a tree based on an ElementNode.

        :param root_node: the root element node.
        :param replace: if `True` the root element is replaced by a document node.         This is usually useful for extended data models (more element children, text nodes).
        """
    etree_module_name = root_node.elem.__class__.__module__
    etree: ModuleType = import_module(etree_module_name)
    assert root_node.elements is not None, 'Not a root element node'
    assert all((not isinstance(x, SchemaElementNode) for x in root_node.elements))
    elements = cast(Dict[ElementProtocol, ElementNode], root_node.elements)
    if replace:
        document = etree.ElementTree()
        if sum((isinstance(x, ElementNode) for x in root_node.children)) == 1:
            for child in root_node.children:
                if isinstance(child, ElementNode):
                    document = etree.ElementTree(child.elem)
                    break
        document_node = cls(document, root_node.uri, root_node.position)
        for child in root_node.children:
            document_node.children.append(child)
            child.parent = document_node
        elements.pop(root_node, None)
        document_node.elements = elements
        del root_node
        return document_node
    else:
        document = etree.ElementTree(root_node.elem)
        document_node = cls(document, root_node.uri, root_node.position - 1)
        document_node.children.append(root_node)
        root_node.parent = document_node
        document_node.elements = elements
    return document_node