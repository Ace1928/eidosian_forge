from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@constructor('integer')
@constructor('nonNegativeInteger')
@constructor('positiveInteger')
@constructor('nonPositiveInteger')
@constructor('negativeInteger')
@constructor('long')
@constructor('int')
@constructor('short')
@constructor('byte')
@constructor('unsignedLong')
@constructor('unsignedInt')
@constructor('unsignedShort')
@constructor('unsignedByte')
def cast_integer_types(self, value):
    try:
        return xsd10_atomic_types[self.symbol](value)
    except ValueError:
        msg = 'could not convert {!r} to xs:{}'.format(value, self.symbol)
        if isinstance(value, (str, bytes, int, UntypedAtomic)):
            raise self.error('FORG0001', msg) from None
        raise self.error('FOCA0002', msg) from None
    except OverflowError as err:
        raise self.error('FOCA0002', err) from None