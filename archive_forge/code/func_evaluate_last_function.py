import math
import decimal
from ..helpers import get_double
from ..datatypes import Duration, DayTimeDuration, YearMonthDuration, \
from ..namespaces import XML_ID, XML_LANG, get_prefixed_name
from ..xpath_nodes import XPathNode, ElementNode, TextNode, CommentNode, \
from ..xpath_tokens import XPathFunction
from ..xpath_context import XPathSchemaContext
from ._xpath1_operators import XPath1Parser
@method(function('last', nargs=0, sequence_types=('xs:integer',)))
def evaluate_last_function(self, context=None):
    if self.context is not None:
        context = self.context
    elif context is None:
        raise self.missing_context()
    return context.size