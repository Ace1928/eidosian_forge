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
@method('{', bp=95)
def nud_namespace_uri(self):
    if self.parser.strict and self.symbol == '{':
        raise self.wrong_syntax('not allowed symbol if parser has strict=True')
    self.parser.next_token.unexpected('{')
    if self.parser.next_token.symbol == '}':
        namespace = ''
    else:
        namespace = self.parser.next_token.value + self.parser.advance_until('}')
        namespace = collapse_white_spaces(namespace)
    try:
        AnyURI(namespace)
    except ValueError as err:
        msg = f'invalid URI in an EQName: {str(err)}'
        raise self.error('XQST0046', msg) from None
    if namespace == XMLNS_NAMESPACE:
        msg = f'cannot use the URI {XMLNS_NAMESPACE!r}!r in an EQName'
        raise self.error('XQST0070', msg)
    self.parser.advance()
    if not self.parser.next_token.label.endswith('function'):
        self.parser.expected_next('(name)', '*')
    self.parser.next_token.bind_namespace(namespace)
    self[:] = (self.parser.symbol_table['(string)'](self.parser, namespace), self.parser.expression(90))
    if self[1].value is None or not self[0].value:
        self.value = self[1].value
    else:
        self.value = '{%s}%s' % (self[0].value, self[1].value)
    return self