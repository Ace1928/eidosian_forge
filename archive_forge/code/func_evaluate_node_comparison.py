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
@method('is')
@method(infix('<<', bp=30))
@method(infix('>>', bp=30))
def evaluate_node_comparison(self, context=None):
    symbol = self.symbol
    left = [x for x in self[0].select(context)]
    if not left:
        return []
    elif len(left) > 1 or not isinstance(left[0], XPathNode):
        raise self[0].error('XPTY0004', 'left operand of %r must be a single node' % symbol)
    right = [x for x in self[1].select(context)]
    if not right:
        return []
    elif len(right) > 1 or not isinstance(right[0], XPathNode):
        raise self[0].error('XPTY0004', 'right operand of %r must be a single node' % symbol)
    if symbol == 'is':
        return left[0] is right[0]
    else:
        if left[0] is right[0]:
            return False
        documents = [context.root]
        documents.extend((v for v in context.variables.values() if isinstance(v, DocumentNode)))
        for root in documents:
            for item in root.iter_document():
                if left[0] is item:
                    return True if symbol == '<<' else False
                elif right[0] is item:
                    return False if symbol == '<<' else True
        else:
            raise self.error('FOCA0002', 'operands are not nodes of the XML tree!')