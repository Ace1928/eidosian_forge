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
def get_xsd_type(self, item: Union[str, PrincipalNodeType]) -> Optional[XsdTypeProtocol]:
    """
        Returns the XSD type associated with an item. Match by item's name
        and XSD validity. Returns `None` if no XSD type is matching.

        :param item: a string or an AttributeNode or an element.
        """
    if not self.xsd_types or isinstance(self.xsd_types, AbstractSchemaProxy):
        return None
    elif isinstance(item, AttributeNode):
        if item.xsd_type is not None:
            return item.xsd_type
        xsd_type = self.xsd_types.get(item.name)
    elif isinstance(item, ElementNode):
        if item.xsd_type is not None:
            return item.xsd_type
        xsd_type = self.xsd_types.get(item.elem.tag)
    elif isinstance(item, str):
        xsd_type = self.xsd_types.get(item)
    else:
        return None
    x: XsdTypeProtocol
    if not xsd_type:
        return None
    elif not isinstance(xsd_type, list):
        return xsd_type
    elif isinstance(item, AttributeNode):
        for x in xsd_type:
            if x.is_valid(item.value):
                return x
    elif isinstance(item, ElementNode):
        for x in xsd_type:
            if x.is_simple():
                if x.is_valid(item.elem.text):
                    return x
            elif x.is_valid(item.elem):
                return x
    return xsd_type[0]