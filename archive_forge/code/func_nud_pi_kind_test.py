import math
import decimal
from ..helpers import get_double
from ..datatypes import Duration, DayTimeDuration, YearMonthDuration, \
from ..namespaces import XML_ID, XML_LANG, get_prefixed_name
from ..xpath_nodes import XPathNode, ElementNode, TextNode, CommentNode, \
from ..xpath_tokens import XPathFunction
from ..xpath_context import XPathSchemaContext
from ._xpath1_operators import XPath1Parser
@method('processing-instruction')
def nud_pi_kind_test(self):
    self.parser.advance('(')
    if self.parser.next_token.symbol != ')':
        self.parser.next_token.expected('(name)', '(string)')
        self[0:] = (self.parser.expression(5),)
    self.parser.advance(')')
    self.value = None
    return self