from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@constructor('decimal')
@constructor('double')
@constructor('float')
def cast_numeric_types(self, value):
    try:
        if self.parser.xsd_version == '1.0':
            return xsd10_atomic_types[self.symbol](value)
        return xsd11_atomic_types[self.symbol](value)
    except ValueError as err:
        if isinstance(value, (str, UntypedAtomic)):
            raise self.error('FORG0001', err)
        raise self.error('FOCA0002', err)