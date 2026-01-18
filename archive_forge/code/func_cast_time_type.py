from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@constructor('time')
def cast_time_type(self, value):
    if isinstance(value, Time):
        return value
    try:
        if isinstance(value, UntypedAtomic):
            return Time.fromstring(value.value)
        elif isinstance(value, DateTime10):
            return Time(value.hour, value.minute, value.second, value.microsecond, value.tzinfo)
        return Time.fromstring(value)
    except ValueError as err:
        raise self.error('FORG0001', err)