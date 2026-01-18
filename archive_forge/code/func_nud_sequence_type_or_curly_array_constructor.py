from ..helpers import iter_sequence
from ..sequence_types import is_sequence_type, match_sequence_type
from ..xpath_tokens import XPathToken, ProxyToken, XPathFunction, XPathMap, XPathArray
from .xpath31_parser import XPath31Parser
@method('array')
def nud_sequence_type_or_curly_array_constructor(self):
    if self.parser.next_token.symbol == '{':
        self.parser.token = XPathArray(self.parser).nud()
        return self.parser.token
    elif self.parser.next_token.symbol != '(':
        return self.as_name()
    self.label = 'kind test'
    self.parser.advance('(')
    if self.parser.next_token.label not in ('kind test', 'function test'):
        self.parser.expected_next('(name)', ':', '*', 'item')
    self[:] = (self.parser.expression(45),)
    if self[0].symbol != '*':
        self[0].parse_occurrence()
    self.parser.advance(')')
    self.parse_occurrence()
    self.value = None
    return self