import math
import decimal
from ..helpers import get_double
from ..datatypes import Duration, DayTimeDuration, YearMonthDuration, \
from ..namespaces import XML_ID, XML_LANG, get_prefixed_name
from ..xpath_nodes import XPathNode, ElementNode, TextNode, CommentNode, \
from ..xpath_tokens import XPathFunction
from ..xpath_context import XPathSchemaContext
from ._xpath1_operators import XPath1Parser
@method(function('concat', nargs=(2, None), sequence_types=('xs:anyAtomicType?', 'xs:anyAtomicType?', 'xs:string')))
def evaluate_concat_function(self, context=None):
    if self.context is not None:
        context = self.context
    return ''.join((self.string_value(self.get_argument(context, index=k)) for k in range(len(self))))