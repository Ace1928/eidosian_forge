from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@method('untypedAtomic')
def evaluate_untyped_atomic(self, context=None):
    arg = self.data_value(self.get_argument(self.context or context))
    if arg is None:
        return []
    elif isinstance(arg, UntypedAtomic):
        return arg
    else:
        return self.cast(arg)