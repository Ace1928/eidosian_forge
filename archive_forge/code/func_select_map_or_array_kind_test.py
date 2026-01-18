from ..helpers import iter_sequence
from ..sequence_types import is_sequence_type, match_sequence_type
from ..xpath_tokens import XPathToken, ProxyToken, XPathFunction, XPathMap, XPathArray
from .xpath31_parser import XPath31Parser
@method('map')
@method('array')
def select_map_or_array_kind_test(self, context=None):
    if context is None:
        raise self.missing_context()
    for item in context.iter_children_or_self():
        if match_sequence_type(item, self.source, self.parser):
            yield item