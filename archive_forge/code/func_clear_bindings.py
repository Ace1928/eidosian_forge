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
def clear_bindings(self) -> None:
    for xsd_element in self.iter_components(xsd_classes=XsdElement):
        assert isinstance(xsd_element, XsdElement)
        xsd_element.binding = None