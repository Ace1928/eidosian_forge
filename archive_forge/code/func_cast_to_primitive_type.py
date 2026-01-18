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
def cast_to_primitive_type(self, obj: Any, type_name: str) -> Any:
    if obj is None or not type_name.startswith('xs:') or type_name.count(':') != 1:
        return obj
    type_name = type_name[3:].rstrip('+*?')
    token = cast(XPathConstructor, self.parser.symbol_table[type_name](self.parser))

    def cast_value(v: Any) -> Any:
        try:
            if isinstance(v, (UntypedAtomic, AnyURI)):
                return token.cast(v)
            elif isinstance(v, float) or isinstance(v, xsd10_atomic_types[XSD_DECIMAL]):
                if type_name in ('double', 'float'):
                    return token.cast(v)
        except (ValueError, TypeError):
            return v
        else:
            return v
    if isinstance(obj, list):
        return [cast_value(x) for x in obj]
    else:
        return cast_value(obj)