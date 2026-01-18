from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@method('NOTATION')
def nud_notation_type(self):
    self.parser.advance('(')
    if self.parser.next_token.symbol == ')':
        raise self.error('XPST0017', 'expected exactly one argument')
    self[0:] = (self.parser.expression(5),)
    if self.parser.next_token.symbol != ')':
        raise self.error('XPST0017', 'expected exactly one argument')
    self.parser.advance()
    self.value = None
    raise self.error('XPST0017', 'no constructor function exists for xs:NOTATION')