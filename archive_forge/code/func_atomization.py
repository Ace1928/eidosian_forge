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
def atomization(self, context: ContextArgType=None) -> Iterator[AtomicValueType]:
    """
        Helper method for value atomization of a sequence.

        Ref: https://www.w3.org/TR/xpath31/#id-atomization

        :param context: the XPath dynamic context.
        """
    for item in self.iter_flatten(context):
        if isinstance(item, XPathNode):
            try:
                value = item.typed_value
            except (TypeError, ValueError) as err:
                raise self.error('XPDY0050', str(err))
            else:
                if value is None:
                    msg = f'argument node {item!r} does not have a typed value'
                    raise self.error('FOTY0012', msg)
                elif isinstance(value, list):
                    yield from value
                else:
                    yield value
        elif isinstance(item, XPathFunction) and (not isinstance(item, XPathArray)):
            raise self.error('FOTY0013', f'{item.label!r} has no typed value')
        elif isinstance(item, AnyAtomicType):
            yield cast(AtomicValueType, item)
        else:
            msg = f'sequence item {item!r} is not appropriate for the context'
            raise self.error('XPTY0004', msg)