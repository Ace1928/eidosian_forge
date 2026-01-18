from ..xpath_nodes import ElementNode
from ..xpath_context import XPathSchemaContext
from ._xpath1_functions import XPath1Parser
@method('@', bp=80)
def nud_attribute_reference(self):
    self.parser.expected_next('*', '(name)', ':', '{', 'Q{', message='invalid attribute specification')
    self[:] = (self.parser.expression(rbp=80),)
    return self