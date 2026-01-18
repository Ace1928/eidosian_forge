import math
import decimal
import operator
from copy import copy
from ..datatypes import AnyURI
from ..exceptions import ElementPathKeyError, ElementPathTypeError
from ..helpers import collapse_white_spaces, node_position
from ..datatypes import AbstractDateTime, Duration, DayTimeDuration, \
from ..xpath_context import XPathSchemaContext
from ..namespaces import XMLNS_NAMESPACE, XSD_NAMESPACE
from ..schema_proxy import AbstractSchemaProxy
from ..xpath_nodes import XPathNode, ElementNode, AttributeNode, DocumentNode
from ..xpath_tokens import XPathToken
from .xpath1_parser import XPath1Parser
@method('|')
def select_union_operator(self, context=None):
    if context is None:
        raise self.missing_context()
    results = {item for k in range(2) for item in self[k].select(copy(context))}
    if any((not isinstance(x, XPathNode) for x in results)):
        raise self.error('XPTY0004', 'only XPath nodes are allowed')
    elif not self.cut_and_sort:
        yield from results
    else:
        yield from sorted(results, key=node_position)