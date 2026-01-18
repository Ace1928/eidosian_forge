import math
import decimal
from ..helpers import get_double
from ..datatypes import Duration, DayTimeDuration, YearMonthDuration, \
from ..namespaces import XML_ID, XML_LANG, get_prefixed_name
from ..xpath_nodes import XPathNode, ElementNode, TextNode, CommentNode, \
from ..xpath_tokens import XPathFunction
from ..xpath_context import XPathSchemaContext
from ._xpath1_operators import XPath1Parser
@method(function('boolean', nargs=1, sequence_types=('item()*', 'xs:boolean')))
def evaluate_boolean_function(self, context=None):
    return self.boolean_value([x for x in self[0].select(self.context or context)])