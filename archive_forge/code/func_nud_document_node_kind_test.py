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
@method('document-node')
def nud_document_node_kind_test(self):
    self.parser.advance('(')
    if self.parser.next_token.symbol in ('element', 'schema-element'):
        self[0:] = (self.parser.expression(5),)
        if self.parser.next_token.symbol == ',':
            msg = 'Too many arguments: expected at most 1 argument'
            raise self.error('XPST0017', msg)
    elif self.parser.next_token.symbol != ')':
        raise self.error('XPST0003', 'element or schema-element kind test expected')
    self.parser.advance(')')
    self.value = None
    return self