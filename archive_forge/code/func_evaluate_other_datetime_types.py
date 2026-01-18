from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@method('date')
@method('gDay')
@method('gMonth')
@method('gMonthDay')
@method('gYear')
@method('gYearMonth')
@method('time')
def evaluate_other_datetime_types(self, context=None):
    if self.context is not None:
        context = self.context
    arg = self.data_value(self.get_argument(context))
    if arg is None:
        return []
    try:
        return self.cast(arg)
    except (TypeError, OverflowError) as err:
        if isinstance(context, XPathSchemaContext):
            return []
        elif isinstance(err, TypeError):
            raise self.error('FORG0006', err) from None
        else:
            raise self.error('FODT0001', err) from None