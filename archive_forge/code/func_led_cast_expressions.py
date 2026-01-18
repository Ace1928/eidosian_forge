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
@method('castable', bp=62)
@method('cast', bp=63)
def led_cast_expressions(self, left):
    self.parser.advance('as')
    self.parser.expected_next('(name)', ':', 'Q{', message='an EQName expected')
    self[:] = (left, self.parser.expression(rbp=85))
    if self.parser.next_token.symbol == '?':
        self[1].occurrence = '?'
        self.parser.advance()
    return self