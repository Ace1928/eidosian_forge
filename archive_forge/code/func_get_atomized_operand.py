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
def get_atomized_operand(self, context: ContextArgType=None) -> Optional[AtomicValueType]:
    """
        Get the atomized value for an XPath operator.

        :param context: the XPath dynamic context.
        :return: the atomized value of a single length sequence or `None` if the sequence is empty.
        """
    selector = iter(self.atomization(context))
    try:
        value = next(selector)
    except StopIteration:
        return None
    else:
        item = getattr(context, 'item', None)
        try:
            next(selector)
        except StopIteration:
            if isinstance(value, UntypedAtomic):
                value = str(value)
            if not isinstance(context, XPathSchemaContext) and item is not None and self.xsd_types and isinstance(value, str):
                xsd_type = self.get_xsd_type(item)
                if xsd_type is None or xsd_type.name in _XSD_SPECIAL_TYPES:
                    pass
                else:
                    try:
                        value = xsd_type.decode(value)
                    except (TypeError, ValueError):
                        msg = 'Type {!r} is not appropriate for the context'
                        raise self.error('XPTY0004', msg.format(type(value)))
            return value
        else:
            msg = 'atomized operand is a sequence of length greater than one'
            raise self.error('XPTY0004', msg)