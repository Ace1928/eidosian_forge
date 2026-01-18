import math
import decimal
from ..helpers import get_double
from ..datatypes import Duration, DayTimeDuration, YearMonthDuration, \
from ..namespaces import XML_ID, XML_LANG, get_prefixed_name
from ..xpath_nodes import XPathNode, ElementNode, TextNode, CommentNode, \
from ..xpath_tokens import XPathFunction
from ..xpath_context import XPathSchemaContext
from ._xpath1_operators import XPath1Parser
@method(function('number', nargs=(0, 1), sequence_types=('xs:anyAtomicType?', 'xs:double')))
def evaluate_number_function(self, context=None):
    arg = self.get_argument(self.context or context, default_to_context=True)
    return self.number_value(arg)