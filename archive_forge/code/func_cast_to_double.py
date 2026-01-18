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
def cast_to_double(self, value: Union[SupportsFloat, str]) -> float:
    """Cast a value to xs:double."""
    try:
        if self.parser.xsd_version == '1.0':
            return cast(float, DoubleProxy10(value))
        return cast(float, DoubleProxy(value))
    except ValueError as err:
        raise self.error('FORG0001', str(err))