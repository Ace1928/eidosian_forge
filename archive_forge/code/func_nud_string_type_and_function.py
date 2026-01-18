from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@method('string')
def nud_string_type_and_function(self):
    try:
        self.parser.advance('(')
        if self.label != 'function' or self.parser.next_token.symbol != ')':
            self[0:] = (self.parser.expression(5),)
        self.parser.advance(')')
    except ElementPathSyntaxError as err:
        raise self.error('XPST0017', err)
    self.value = None
    return self