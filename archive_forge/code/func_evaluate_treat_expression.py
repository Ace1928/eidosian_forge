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
@method('treat')
def evaluate_treat_expression(self, context=None):
    occurs = self[1].occurrence
    position = None
    castable_expr = []
    if self[1].symbol == 'empty-sequence':
        for _ in self[0].select(context):
            raise self.error('XPDY0050')
    elif self[1].label in ('kind test', 'sequence type', 'function test'):
        for position, item in enumerate(self[0].select(context)):
            result = self[1].evaluate(context)
            if isinstance(result, list) and (not result):
                raise self.error('XPDY0050')
            elif position and (occurs is None or occurs == '?'):
                raise self.error('XPDY0050', 'more than one item in sequence')
            castable_expr.append(item)
        else:
            if position is None and occurs not in ('*', '?'):
                raise self.error('XPDY0050', 'the sequence cannot be empty')
    else:
        type_name = self[1].source.rstrip('*+?')
        try:
            qname = get_expanded_name(type_name, self.parser.namespaces)
        except KeyError as err:
            raise self.error('XPST0081', 'prefix {} not found'.format(str(err)))
        if not qname.startswith('{') and (not QName.is_valid(qname)):
            raise self.error('XPST0003')
        for position, item in enumerate(self[0].select(context)):
            try:
                if not is_instance(item, qname, self.parser):
                    msg = f'item {item!r} is not of type {type_name!r}'
                    raise self.error('XPDY0050', msg)
            except KeyError:
                msg = f'atomic type {type_name!r} not found in in-scope schema types'
                raise self.error('XPST0051', msg) from None
            else:
                if position and (occurs is None or occurs == '?'):
                    raise self.error('XPDY0050', 'more than one item in sequence')
                castable_expr.append(item)
        else:
            if position is None and occurs not in ('*', '?'):
                raise self.error('XPDY0050', 'the sequence cannot be empty')
    return castable_expr