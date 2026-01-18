from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@method('dateTimeStamp')
def evaluate_datetime_stamp_type(self, context=None):
    if self.context is not None:
        context = self.context
    arg = self.data_value(self.get_argument(context))
    if arg is None:
        return []
    if isinstance(arg, UntypedAtomic):
        return self.cast(arg.value)
    elif isinstance(arg, Date):
        return self.cast(arg)
    return self.cast(str(arg))