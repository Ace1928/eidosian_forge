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
@method(infix('+', bp=40))
def evaluate_plus_operator(self, context=None):
    if len(self) == 1:
        arg = self.get_argument(context, cls=NumericProxy)
        return [] if arg is None else +arg
    else:
        op1, op2 = self.get_operands(context, cls=ArithmeticProxy)
        if op1 is None:
            return []
        try:
            return op1 + op2
        except (TypeError, OverflowError) as err:
            if isinstance(context, XPathSchemaContext):
                return []
            elif isinstance(err, TypeError):
                raise self.error('XPTY0004', err) from None
            elif isinstance(op1, AbstractDateTime):
                raise self.error('FODT0001', err) from None
            elif isinstance(op1, Duration):
                raise self.error('FODT0002', err) from None
            else:
                raise self.error('FOAR0002', err) from None