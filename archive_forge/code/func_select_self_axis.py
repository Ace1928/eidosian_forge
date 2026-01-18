from ..xpath_nodes import ElementNode
from ..xpath_context import XPathSchemaContext
from ._xpath1_functions import XPath1Parser
@method(axis('self'))
def select_self_axis(self, context=None):
    if context is None:
        raise self.missing_context()
    else:
        for _ in context.iter_self():
            yield from self[0].select(context)