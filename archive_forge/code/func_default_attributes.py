from typing import cast, Any, Callable, Iterator, List, Optional, Tuple, Union
from elementpath.datatypes import AnyAtomicType
from ..exceptions import XMLSchemaValueError
from ..names import XSD_GROUP, XSD_ATTRIBUTE_GROUP, XSD_SEQUENCE, XSD_OVERRIDE, \
from ..aliases import ElementType, NamespacesType, SchemaType, ComponentClassType, \
from ..translation import gettext as _
from ..helpers import get_qname, local_name
from .exceptions import XMLSchemaDecodeError
from .helpers import get_xsd_derivation_attribute
from .xsdbase import XSD_TYPE_DERIVATIONS, XsdComponent, XsdType, ValidationMixin
from .attributes import XsdAttributeGroup
from .assertions import XsdAssert
from .simple_types import FacetsValueType, XsdSimpleType, XsdUnion
from .groups import XsdGroup
from .wildcards import XsdOpenContent, XsdDefaultOpenContent
@property
def default_attributes(self) -> Optional[XsdAttributeGroup]:
    if self.redefine is not None:
        default_attributes = self.schema.default_attributes
    else:
        for child in self.schema.root:
            if child.tag == XSD_OVERRIDE and self.elem in child:
                schema = self.schema.includes[child.attrib['schemaLocation']]
                if schema.override is self.schema:
                    default_attributes = schema.default_attributes
                    break
        else:
            default_attributes = self.schema.default_attributes
    if isinstance(default_attributes, str):
        return None
    return default_attributes