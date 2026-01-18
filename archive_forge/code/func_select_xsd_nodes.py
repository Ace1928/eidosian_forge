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
def select_xsd_nodes(self, schema_context: XPathSchemaContext, name: str) -> Iterator[Union[None, AttributeNode, ElementNode]]:
    """
        Selector for XSD nodes (elements, attributes and schemas). If there is
        a match with an attribute or an element the node's type is added to
        matching types of the token. For each matching elements or attributes
        yields tuple nodes containing the node, its type and a compatible value
        for doing static evaluation. For matching schemas yields the original
        instance.

        :param schema_context: an XPathSchemaContext instance.
        :param name: a QName in extended format.
        """
    xsd_node: Any
    xsd_root = cast(Union[XsdSchemaProtocol, XsdElementProtocol], schema_context.root.value)
    for xsd_node in schema_context.iter_children_or_self():
        if xsd_node is None:
            if name == XSD_SCHEMA == schema_context.root.elem.tag:
                yield None
        elif isinstance(xsd_node, AttributeNode):
            assert not isinstance(xsd_node.value, str)
            if not xsd_node.value.is_matching(name):
                continue
            if xsd_node.name is not None:
                self.add_xsd_type(xsd_node)
            else:
                xsd_attribute = xsd_root.maps.attributes.get(name)
                if xsd_attribute is not None:
                    self.add_xsd_type(xsd_attribute)
            yield xsd_node
        elif isinstance(xsd_node, SchemaElementNode):
            if name == XSD_SCHEMA == xsd_node.elem.tag:
                yield xsd_node
            elif xsd_node.elem.is_matching(name, self.parser.namespaces.get('')):
                if xsd_node.elem.name is not None:
                    self.add_xsd_type(xsd_node)
                else:
                    xsd_element = xsd_root.maps.elements.get(name)
                    if xsd_element is not None:
                        for child in schema_context.root.children:
                            if child.value is xsd_element:
                                xsd_node = child
                                self.add_xsd_type(xsd_node)
                                break
                        else:
                            self.add_xsd_type(xsd_element)
                yield xsd_node