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
@method(register('(name)', bp=10, label='literal'))
def nud_name_literal(self):
    if self.parser.next_token.symbol == '::':
        msg = "axis '%s::' not found" % self.value
        if self.parser.compatibility_mode:
            raise self.error('XPST0010', msg)
        raise self.error('XPST0003', msg)
    elif self.parser.next_token.symbol == '(':
        if self.parser.version >= '2.0':
            pass
        elif self.namespace == XSD_NAMESPACE:
            raise self.error('XPST0017', 'unknown constructor function {!r}'.format(self.value))
        elif self.namespace or self.value not in self.parser.RESERVED_FUNCTION_NAMES:
            raise self.error('XPST0017', 'unknown function {!r}'.format(self.value))
        else:
            msg = f'{self.value!r} is not allowed as function name'
            raise self.error('XPST0003', msg)
    return self