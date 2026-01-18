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
@method('if', bp=20)
def nud_if_expression(self):
    if self.parser.next_token.symbol != '(':
        return self.as_name()
    self.parser.advance('(')
    self[:] = (self.parser.expression(5),)
    self.parser.advance(')')
    self.parser.advance('then')
    self[1:] = (self.parser.expression(5),)
    self.parser.advance('else')
    self[2:] = (self.parser.expression(5),)
    return self