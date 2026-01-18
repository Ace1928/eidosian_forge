import math
import operator
from copy import copy
from decimal import Decimal, DivisionByZero
from ..exceptions import ElementPathError
from ..helpers import OCCURRENCE_INDICATORS, numeric_equal, numeric_not_equal, \
from ..namespaces import XSD_NAMESPACE, XSD_NOTATION, XSD_ANY_ATOMIC_TYPE, \
from ..datatypes import get_atomic_value, UntypedAtomic, QName, AnyURI, \
from ..xpath_nodes import ElementNode, DocumentNode, XPathNode, AttributeNode
from ..sequence_types import is_instance
from ..xpath_context import XPathSchemaContext
from ..xpath_tokens import XPathFunction
from .xpath2_parser import XPath2Parser
@method('eq')
@method('ne')
@method('lt')
@method('gt')
@method('le')
@method('ge')
def evaluate_value_comparison_operators(self, context=None):
    operands = [self[0].get_atomized_operand(context=copy(context)), self[1].get_atomized_operand(context=copy(context))]
    if any((x is None for x in operands)):
        return []
    elif any((isinstance(x, XPathFunction) for x in operands)):
        raise self.error('FOTY0013', 'cannot compare a function item')
    elif all((isinstance(x, DoubleProxy10) for x in operands)):
        if self.symbol == 'eq':
            return numeric_equal(*operands)
        elif self.symbol == 'ne':
            return numeric_not_equal(*operands)
        elif numeric_equal(*operands):
            return self.symbol in ('le', 'ge')
    cls0, cls1 = (type(operands[0]), type(operands[1]))
    if cls0 is cls1 and cls0 is not Duration:
        pass
    elif all((isinstance(x, float) for x in operands)):
        pass
    elif any((isinstance(x, bool) for x in operands)):
        msg = 'cannot apply {} between {!r} and {!r}'.format(self, *operands)
        raise self.error('XPTY0004', msg)
    elif all((isinstance(x, (int, Decimal)) for x in operands)):
        pass
    elif all((isinstance(x, (str, UntypedAtomic, AnyURI)) for x in operands)):
        pass
    elif all((isinstance(x, (str, UntypedAtomic, QName)) for x in operands)):
        pass
    elif all((isinstance(x, (float, Decimal, int)) for x in operands)):
        if isinstance(operands[0], float):
            operands[1] = get_double(operands[1], self.parser.xsd_version)
        else:
            operands[0] = get_double(operands[0], self.parser.xsd_version)
    elif all((isinstance(x, Duration) for x in operands)) and self.symbol in ('eq', 'ne'):
        pass
    elif (issubclass(cls0, cls1) or issubclass(cls1, cls0)) and (not issubclass(cls0, Duration)):
        pass
    else:
        msg = 'cannot apply {} between {!r} and {!r}'.format(self, *operands)
        raise self.error('XPTY0004', msg)
    try:
        return getattr(operator, self.symbol)(*operands)
    except TypeError as err:
        raise self.error('XPTY0004', err) from None