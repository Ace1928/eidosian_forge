from ..xpath_nodes import ElementNode
from ..xpath_context import XPathSchemaContext
from ._xpath1_functions import XPath1Parser
@method('@')
@method(axis('attribute'))
def select_attribute_reference_or_axis(self, context=None):
    if context is None:
        raise self.missing_context()
    else:
        for _ in context.iter_attributes():
            yield from self[0].select(context)