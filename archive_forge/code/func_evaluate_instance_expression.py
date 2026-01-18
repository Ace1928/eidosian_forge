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
@method('instance')
def evaluate_instance_expression(self, context=None):
    occurs = self[1].occurrence
    position = None
    if self[1].symbol == 'empty-sequence':
        for _ in self[0].select(context):
            return False
        return True
    elif self[1].label in ('kind test', 'sequence type', 'function test'):
        if context is None:
            raise self.missing_context()
        context = copy(context)
        for position, context.item in enumerate(self[0].select(context)):
            if context.axis is None:
                context.axis = 'self'
            result = self[1].evaluate(context)
            if isinstance(result, list) and (not result):
                return occurs in ('*', '?')
            elif position and (occurs is None or occurs == '?'):
                return False
        else:
            return position is not None or occurs in ('*', '?')
    else:
        type_name = self[1].source.rstrip('*+?')
        try:
            qname = get_expanded_name(type_name, self.parser.namespaces)
        except KeyError as err:
            raise self.error('XPST0081', 'namespace prefix {} not found'.format(err))
        for position, item in enumerate(self[0].select(context)):
            try:
                if not is_instance(item, qname, self.parser):
                    return False
            except KeyError:
                msg = f'atomic type {type_name!r} not found in in-scope schema types'
                raise self.error('XPST0051', msg) from None
            else:
                if position and (occurs is None or occurs == '?'):
                    return False
        else:
            return position is not None or occurs in ('*', '?')