import math
import decimal
from ..helpers import get_double
from ..datatypes import Duration, DayTimeDuration, YearMonthDuration, \
from ..namespaces import XML_ID, XML_LANG, get_prefixed_name
from ..xpath_nodes import XPathNode, ElementNode, TextNode, CommentNode, \
from ..xpath_tokens import XPathFunction
from ..xpath_context import XPathSchemaContext
from ._xpath1_operators import XPath1Parser
@method(function('round', nargs=1, sequence_types=('xs:numeric?', 'xs:numeric?')))
def evaluate_round_function(self, context=None):
    if self.context is not None:
        context = self.context
    arg = self.get_argument(context)
    if arg is None:
        return math.nan if self.parser.version == '1.0' else []
    elif isinstance(arg, XPathNode) or self.parser.compatibility_mode:
        arg = self.number_value(arg)
    if isinstance(arg, float) and (math.isnan(arg) or math.isinf(arg)):
        return arg
    try:
        number = decimal.Decimal(arg)
        if number > 0:
            return type(arg)(number.quantize(decimal.Decimal('1'), rounding='ROUND_HALF_UP'))
        else:
            return type(arg)(number.quantize(decimal.Decimal('1'), rounding='ROUND_HALF_DOWN'))
    except TypeError as err:
        if isinstance(context, XPathSchemaContext):
            return []
        raise self.error('FORG0006', err) from None
    except decimal.InvalidOperation:
        if not isinstance(arg, str):
            return round(arg)
        elif isinstance(context, XPathSchemaContext):
            return []
        raise self.error('XPTY0004') from None
    except decimal.DecimalException as err:
        if isinstance(context, XPathSchemaContext):
            return []
        raise self.error('FOCA0002', err) from None