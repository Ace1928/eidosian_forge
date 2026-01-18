import warnings
from collections import Counter
from functools import lru_cache
from typing import cast, Any, Callable, Dict, List, Iterable, Iterator, \
from ..exceptions import XMLSchemaKeyError, XMLSchemaTypeError, \
from ..names import XSD_NAMESPACE, XSD_REDEFINE, XSD_OVERRIDE, XSD_NOTATION, \
from ..aliases import ComponentClassType, ElementType, SchemaType, BaseXsdType, \
from ..helpers import get_qname, local_name, get_extended_qname
from ..namespaces import NamespaceResourcesMap
from ..translation import gettext as _
from .exceptions import XMLSchemaNotBuiltError, XMLSchemaModelError, XMLSchemaModelDepthError, \
from .xsdbase import XsdValidator, XsdComponent
from .builtins import xsd_builtin_types_factory
from .models import check_model
from . import XsdAttribute, XsdSimpleType, XsdComplexType, XsdElement, XsdAttributeGroup, \
def lookup_notation(self, qname: str) -> XsdNotation:
    try:
        obj = self.notations[qname]
    except KeyError:
        raise XMLSchemaKeyError(f'xs:notation {qname!r} not found')
    else:
        if isinstance(obj, XsdNotation):
            return obj
        return cast(XsdNotation, self._build_global(obj, qname, self.notations))