from decimal import DecimalException
from typing import cast, Any, Callable, Dict, Iterator, List, \
from xml.etree import ElementTree
from ..aliases import ElementType, AtomicValueType, ComponentClassType, \
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..names import XSD_NAMESPACE, XSD_ANY_TYPE, XSD_SIMPLE_TYPE, XSD_PATTERN, \
from ..translation import gettext as _
from ..helpers import local_name
from .exceptions import XMLSchemaValidationError, XMLSchemaEncodeError, \
from .xsdbase import XsdComponent, XsdType, ValidationMixin
from .facets import XsdFacet, XsdWhiteSpaceFacet, XsdPatternFacets, \
def is_dynamic_consistent(self, other: Any) -> bool:
    return other.name in {XSD_ANY_TYPE, XSD_ANY_SIMPLE_TYPE} or other.is_derived(self) or (isinstance(other, self.__class__) and any((mt1.is_derived(mt2) for mt1 in other.member_types for mt2 in self.member_types)))