import math
import decimal
from ..helpers import get_double
from ..datatypes import Duration, DayTimeDuration, YearMonthDuration, \
from ..namespaces import XML_ID, XML_LANG, get_prefixed_name
from ..xpath_nodes import XPathNode, ElementNode, TextNode, CommentNode, \
from ..xpath_tokens import XPathFunction
from ..xpath_context import XPathSchemaContext
from ._xpath1_operators import XPath1Parser
@method(function('node', nargs=0, label='kind test'))
def select_node_kind_test(self, context=None):
    if context is None:
        raise self.missing_context()
    for item in context.iter_children_or_self():
        if isinstance(item, XPathNode):
            if not isinstance(item, DocumentNode) or item is context.root:
                yield item