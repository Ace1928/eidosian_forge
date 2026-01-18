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
@method(function('element', nargs=(0, 2), label='kind test'))
def select_element_kind_test(self, context=None):
    if context is None:
        raise self.missing_context()
    elif not self:
        for item in context.iter_children_or_self():
            if isinstance(item, ElementNode):
                yield item
    else:
        for item in self[0].select(context):
            if len(self) == 1:
                yield item
            elif isinstance(item, ElementNode):
                try:
                    type_annotation = get_expanded_name(self[1].source, self.parser.namespaces)
                except KeyError:
                    type_annotation = self[1].source
                if item.nilled:
                    if type_annotation[-1] in '*?':
                        yield item
                elif item.xsd_type is not None:
                    if type_annotation == item.xsd_type.name:
                        yield item
                    elif is_instance(item.typed_value, type_annotation, self.parser):
                        yield item
                elif type_annotation == XSD_UNTYPED and self[0].symbol != '*':
                    yield item