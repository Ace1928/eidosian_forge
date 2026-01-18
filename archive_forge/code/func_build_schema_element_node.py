from typing import cast, Any, Iterator, List, Optional, Union
from .namespaces import NamespacesType
from .exceptions import ElementPathTypeError
from .protocols import ElementProtocol, LxmlElementProtocol, \
from .etree import is_etree_document, is_etree_element
from .xpath_nodes import SchemaElemType, ChildNodeType, ElementMapType, \
def build_schema_element_node() -> SchemaElementNode:
    nonlocal position
    node = SchemaElementNode(elem, parent, position, elem.namespaces)
    position += 1
    _elements[elem] = node
    position += len(elem.namespaces) + int('xml' not in elem.namespaces) + len(elem.attrib)
    return node