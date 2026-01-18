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
@method('|', bp=50)
def led_union_operator(self, left):
    self.cut_and_sort = True
    if left.symbol in ('|', 'union'):
        left.cut_and_sort = False
    self[:] = (left, self.parser.expression(rbp=50))
    return self