from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@constructor('gYearMonth')
def cast_gregorian_year_month_type(self, value):
    cls = GregorianYearMonth if self.parser.xsd_version == '1.1' else GregorianYearMonth10
    if isinstance(value, cls):
        return value
    try:
        if isinstance(value, UntypedAtomic):
            return cls.fromstring(value.value)
        elif isinstance(value, (Date10, DateTime10)):
            return cls(value.year, value.month, value.tzinfo)
        return cls.fromstring(value)
    except OverflowError as err:
        raise self.error('FODT0001', err) from None
    except ValueError as err:
        raise self.error('FORG0001', err)