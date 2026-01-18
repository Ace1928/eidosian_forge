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
@method('[')
def select_predicate(self, context=None):
    if context is None:
        raise self.missing_context()
    for _ in context.inner_focus_select(self[0]):
        if (self[1].label in ('axis', 'kind test') or self[1].symbol == '..') and (not isinstance(context.item, XPathNode)):
            raise self.error('XPTY0020')
        elif False and isinstance(context, XPathSchemaContext):
            yield context.item
            continue
        predicate = [x for x in self[1].select(copy(context))]
        if len(predicate) == 1 and isinstance(predicate[0], NumericProxy):
            if context.position == predicate[0]:
                yield context.item
        elif self.boolean_value(predicate):
            yield context.item