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
@method('if')
def select_if_expression(self, context=None):
    if self.boolean_value([x for x in self[0].select(copy(context))]):
        if isinstance(context, XPathSchemaContext):
            self[2].evaluate(copy(context))
        yield from self[1].select(context)
    else:
        if isinstance(context, XPathSchemaContext):
            self[1].evaluate(copy(context))
        yield from self[2].select(context)