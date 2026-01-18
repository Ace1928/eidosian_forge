import math
import decimal
import operator
from copy import copy
from ..datatypes import AnyURI
from ..exceptions import ElementPathKeyError, ElementPathTypeError
from ..helpers import collapse_white_spaces, node_position
from ..datatypes import AbstractDateTime, Duration, DayTimeDuration, \
from ..xpath_context import XPathSchemaContext
from ..namespaces import XMLNS_NAMESPACE, XSD_NAMESPACE
from ..schema_proxy import AbstractSchemaProxy
from ..xpath_nodes import XPathNode, ElementNode, AttributeNode, DocumentNode
from ..xpath_tokens import XPathToken
from .xpath1_parser import XPath1Parser
@method(infix('*', bp=45))
def evaluate_multiply_operator(self, context=None):
    if self:
        op1, op2 = self.get_operands(context, cls=ArithmeticProxy)
        if op1 is None:
            return []
        try:
            if isinstance(op2, (YearMonthDuration, DayTimeDuration)):
                return op2 * op1
            return op1 * op2
        except TypeError as err:
            if isinstance(context, XPathSchemaContext):
                return []
            if isinstance(op1, (float, decimal.Decimal)):
                if math.isnan(op1):
                    raise self.error('FOCA0005') from None
                elif math.isinf(op1):
                    raise self.error('FODT0002') from None
            if isinstance(op2, (float, decimal.Decimal)):
                if math.isnan(op2):
                    raise self.error('FOCA0005') from None
                elif math.isinf(op2):
                    raise self.error('FODT0002') from None
            raise self.error('XPTY0004', err) from None
        except ValueError as err:
            if isinstance(context, XPathSchemaContext):
                return []
            raise self.error('FOCA0005', err) from None
        except OverflowError as err:
            if isinstance(context, XPathSchemaContext):
                return []
            elif isinstance(op1, AbstractDateTime):
                raise self.error('FODT0001', err) from None
            elif isinstance(op1, Duration):
                raise self.error('FODT0002', err) from None
            else:
                raise self.error('FOAR0002', err) from None
    else:
        return [x for x in self.select(context)]