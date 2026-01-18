import math
import operator
from copy import copy
from decimal import Decimal, DivisionByZero
from ..exceptions import ElementPathError
from ..helpers import OCCURRENCE_INDICATORS, numeric_equal, numeric_not_equal, \
from ..namespaces import XSD_NAMESPACE, XSD_NOTATION, XSD_ANY_ATOMIC_TYPE, \
from ..datatypes import get_atomic_value, UntypedAtomic, QName, AnyURI, \
from ..xpath_nodes import ElementNode, DocumentNode, XPathNode, AttributeNode
from ..sequence_types import is_instance
from ..xpath_context import XPathSchemaContext
from ..xpath_tokens import XPathFunction
from .xpath2_parser import XPath2Parser
@method('castable')
@method('cast')
def evaluate_cast_expressions(self, context=None):
    type_name = self[1].source.rstrip('+*?')
    try:
        atomic_type = get_expanded_name(type_name, self.parser.namespaces)
    except KeyError as err:
        raise self.error('XPST0081', 'prefix {} not found'.format(str(err)))
    if atomic_type in (XSD_NOTATION, XSD_ANY_ATOMIC_TYPE):
        raise self.error('XPST0080')
    namespace = get_namespace(atomic_type)
    if namespace != XSD_NAMESPACE and (self.parser.schema is None or self.parser.schema.get_type(atomic_type) is None):
        msg = 'atomic type %r not found in the in-scope schema types'
        raise self.error('XPST0051', msg % atomic_type)
    result = [res for res in self[0].select(context)]
    if len(result) > 1:
        if self.symbol != 'cast':
            return False
        raise self.error('XPTY0004', 'more than one value in expression')
    elif not result:
        if self[1].occurrence == '?':
            return [] if self.symbol == 'cast' else True
        elif self.symbol != 'cast':
            return False
        else:
            raise self.error('XPTY0004', 'an atomic value is required')
    arg = self.data_value(result[0])
    try:
        if namespace != XSD_NAMESPACE:
            value = self.parser.schema.cast_as(self.string_value(arg), atomic_type)
        else:
            local_name = atomic_type.split('}')[1]
            token_class = self.parser.symbol_table.get(local_name)
            if token_class is None or token_class.label != 'constructor function':
                msg = f'atomic type {type_name!r} not found in the in-scope schema types'
                raise self.error('XPST0051', msg)
            elif local_name == 'QName':
                if isinstance(arg, QName):
                    pass
                elif self.parser.version < '3.0' and self[0].symbol != '(string)':
                    raise self.error('XPTY0004', 'Non literal string to QName cast')
            token = token_class(self.parser)
            value = token.cast(arg)
    except ElementPathError:
        if self.symbol != 'cast':
            return False
        elif isinstance(context, XPathSchemaContext):
            return UntypedAtomic('1')
        raise
    except (TypeError, ValueError) as err:
        if self.symbol != 'cast':
            return False
        elif isinstance(context, XPathSchemaContext):
            return UntypedAtomic('1')
        elif isinstance(arg, (UntypedAtomic, str)):
            raise self.error('FORG0001', err) from None
        raise self.error('XPTY0004', err) from None
    else:
        return value if self.symbol == 'cast' else True