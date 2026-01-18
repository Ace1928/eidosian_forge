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
def iter_comparison_data(self, context: ContextArgType) -> Iterator[OperandsType]:
    """
        Generates comparison data couples for the general comparison of sequences.
        Different sequences maybe generated with an XPath 2.0 parser, depending on
        compatibility mode setting.

        Ref: https://www.w3.org/TR/xpath20/#id-general-comparisons

        :param context: the XPath dynamic context.
        """
    left_values: Any
    right_values: Any
    if self.parser.compatibility_mode:
        left_values = [x for x in self._items[0].atomization(copy(context))]
        right_values = [x for x in self._items[1].atomization(copy(context))]
        try:
            if isinstance(left_values[0], bool):
                if len(left_values) == 1:
                    yield (left_values[0], self.boolean_value(right_values))
                    return
            if isinstance(right_values[0], bool):
                if len(right_values) == 1:
                    yield (self.boolean_value(left_values), right_values[0])
                    return
        except IndexError:
            return
        if self.symbol in ('<', '<=', '>', '>='):
            yield from product(map(float, left_values), map(float, right_values))
            return
        elif self.parser.version == '1.0':
            yield from product(left_values, right_values)
            return
    else:
        left_values = self._items[0].atomization(copy(context))
        right_values = self._items[1].atomization(copy(context))
    for values in product(left_values, right_values):
        if any((isinstance(x, bool) for x in values)):
            if any((isinstance(x, (str, Integer)) for x in values)):
                msg = 'cannot compare {!r} and {!r}'
                raise TypeError(msg.format(type(values[0]), type(values[1])))
        elif any((isinstance(x, Integer) for x in values)) and any((isinstance(x, str) for x in values)):
            msg = 'cannot compare {!r} and {!r}'
            raise TypeError(msg.format(type(values[0]), type(values[1])))
        elif any((isinstance(x, float) for x in values)):
            if isinstance(values[0], decimal.Decimal):
                yield (float(values[0]), values[1])
                continue
            elif isinstance(values[1], decimal.Decimal):
                yield (values[0], float(values[1]))
                continue
        yield values