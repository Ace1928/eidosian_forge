import math
import operator
from copy import copy
from decimal import Decimal, DivisionByZero
from ..exceptions import ElementPathError
from ..helpers import OCCURRENCE_INDICATORS, numeric_equal, numeric_not_equal, \
from ..namespaces import XSD_NAMESPACE, XSD_NOTATION, XSD_ANY_ATOMIC_TYPE, \
from ..datatypes import get_atomic_value, UntypedAtomic, QName, AnyURI, \
from ..xpath_nodes import ElementNode, DocumentNode, XPathNode, AttributeNode
from ..sequence_types import is_instance
from ..xpath_context import XPathSchemaContext
from ..xpath_tokens import XPathFunction
from .xpath2_parser import XPath2Parser
@method('eq', bp=30)
@method('ne', bp=30)
@method('lt', bp=30)
@method('gt', bp=30)
@method('le', bp=30)
@method('ge', bp=30)
def led_value_comparison_operators(self, left):
    if left.symbol in COMPARISON_OPERATORS:
        raise self.wrong_syntax()
    self[:] = (left, self.parser.expression(rbp=30))
    return self