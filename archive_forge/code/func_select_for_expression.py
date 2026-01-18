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
@method('for')
def select_for_expression(self, context=None):
    if context is None:
        raise self.missing_context()
    context = copy(context)
    varnames = [self[k][0].value for k in range(0, len(self) - 1, 2)]
    selectors = [self[k].select for k in range(1, len(self) - 1, 2)]
    for results in copy(context).iter_product(selectors, varnames):
        context.variables.update((x for x in zip(varnames, results)))
        yield from self[-1].select(copy(context))