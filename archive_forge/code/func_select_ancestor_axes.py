from ..xpath_nodes import ElementNode
from ..xpath_context import XPathSchemaContext
from ._xpath1_functions import XPath1Parser
@method(axis('ancestor', reverse_axis=True))
@method(axis('ancestor-or-self', reverse_axis=True))
def select_ancestor_axes(self, context=None):
    if context is None:
        raise self.missing_context()
    else:
        for _ in context.iter_ancestors(axis=self.symbol):
            yield from self[0].select(context)