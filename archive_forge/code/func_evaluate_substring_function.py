import math
import decimal
from ..helpers import get_double
from ..datatypes import Duration, DayTimeDuration, YearMonthDuration, \
from ..namespaces import XML_ID, XML_LANG, get_prefixed_name
from ..xpath_nodes import XPathNode, ElementNode, TextNode, CommentNode, \
from ..xpath_tokens import XPathFunction
from ..xpath_context import XPathSchemaContext
from ._xpath1_operators import XPath1Parser
@method(function('substring', nargs=(2, 3), sequence_types=('xs:string?', 'xs:double', 'xs:double', 'xs:string')))
def evaluate_substring_function(self, context=None):
    if self.context is not None:
        context = self.context
    item = self.get_argument(context, default='', cls=str)
    try:
        start = self.get_argument(context, index=1, required=True)
        if math.isnan(start) or math.isinf(start):
            return ''
    except TypeError:
        if isinstance(context, XPathSchemaContext):
            start = 0
        else:
            raise self.error('FORG0006', 'the second argument must be xs:numeric') from None
    else:
        start = int(round(start)) - 1
    if len(self) == 2:
        return item[max(start, 0):]
    else:
        try:
            length = self.get_argument(context, index=2, required=True)
            if math.isnan(length) or length <= 0:
                return ''
        except TypeError:
            if isinstance(context, XPathSchemaContext):
                length = len(item)
            else:
                raise self.error('FORG0006', 'the third argument must be xs:numeric') from None
        if math.isinf(length):
            return item[max(start, 0):]
        else:
            stop = start + int(round(length))
            return item[slice(max(start, 0), max(stop, 0))]