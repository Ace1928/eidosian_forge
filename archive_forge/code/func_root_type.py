import re
from typing import TYPE_CHECKING, cast, Any, Dict, Generic, List, Iterator, Optional, \
from xml.etree import ElementTree
from elementpath import select
from elementpath.etree import is_etree_element, etree_tostring
from ..exceptions import XMLSchemaValueError, XMLSchemaTypeError
from ..names import XSD_ANNOTATION, XSD_APPINFO, XSD_DOCUMENTATION, \
from ..aliases import ElementType, NamespacesType, SchemaType, BaseXsdType, \
from ..translation import gettext as _
from ..helpers import get_qname, local_name, get_prefixed_qname
from ..resources import XMLResource
from .exceptions import XMLSchemaParseError, XMLSchemaValidationError
@property
def root_type(self) -> BaseXsdType:
    """
        The root type of the type definition hierarchy. For an atomic type
        is the primitive type. For a list is the primitive type of the item.
        For a union is the base union type. For a complex type is xs:anyType.
        """
    if getattr(self, 'attributes', None):
        return cast('XsdComplexType', self.maps.types[XSD_ANY_TYPE])
    elif self.base_type is None:
        if self.is_simple():
            return cast('XsdSimpleType', self)
        return cast('XsdComplexType', self.maps.types[XSD_ANY_TYPE])
    primitive_type: BaseXsdType
    try:
        if self.base_type.is_simple():
            primitive_type = self.base_type.primitive_type
        else:
            primitive_type = self.base_type.content.primitive_type
    except AttributeError:
        return self.base_type.root_type
    else:
        return primitive_type