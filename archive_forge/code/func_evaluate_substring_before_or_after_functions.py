import math
import decimal
from ..helpers import get_double
from ..datatypes import Duration, DayTimeDuration, YearMonthDuration, \
from ..namespaces import XML_ID, XML_LANG, get_prefixed_name
from ..xpath_nodes import XPathNode, ElementNode, TextNode, CommentNode, \
from ..xpath_tokens import XPathFunction
from ..xpath_context import XPathSchemaContext
from ._xpath1_operators import XPath1Parser
@method(function('substring-before', nargs=2, sequence_types=('xs:string?', 'xs:string?', 'xs:string')))
@method(function('substring-after', nargs=2, sequence_types=('xs:string?', 'xs:string?', 'xs:string')))
def evaluate_substring_before_or_after_functions(self, context=None):
    if self.context is not None:
        context = self.context
    arg1 = self.get_argument(context, default='', cls=str)
    arg2 = self.get_argument(context, index=1, default='', cls=str)
    index = arg1.find(arg2)
    if index < 0:
        return ''
    if self.symbol == 'substring-before':
        return arg1[:index]
    else:
        return arg1[index + len(arg2):]