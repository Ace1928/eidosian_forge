import warnings
from collections.abc import MutableMapping
from copy import copy as _copy
from typing import TYPE_CHECKING, cast, overload, Any, Iterable, Iterator, \
from xml.etree import ElementTree
from .. import limits
from ..exceptions import XMLSchemaValueError
from ..names import XSD_GROUP, XSD_SEQUENCE, XSD_ALL, XSD_CHOICE, XSD_ELEMENT, \
from ..aliases import ElementType, NamespacesType, SchemaType, IterDecodeType, \
from ..translation import gettext as _
from ..helpers import get_qname, local_name, raw_xml_encode
from ..converters import ElementData
from .exceptions import XMLSchemaModelError, XMLSchemaModelDepthError, \
from .xsdbase import ValidationMixin, XsdComponent, XsdType
from .particles import ParticleMixin, OccursCalculator
from .elements import XsdElement, XsdAlternative
from .wildcards import XsdAnyElement, Xsd11AnyElement
from .models import ModelVisitor, iter_unordered_content, iter_collapsed_content
def is_element_restriction(self, other: ModelParticleType) -> bool:
    if self.xsd_version == '1.0' and isinstance(other, XsdElement) and (not other.ref) and (other.name not in self.schema.substitution_groups):
        return False
    elif not self.has_occurs_restriction(other):
        return False
    elif self.model == 'choice':
        if other.name in self.maps.substitution_groups and all((isinstance(e, XsdElement) and e.substitution_group == other.name for e in self)):
            return True
        return any((e.is_restriction(other, False) for e in self))
    else:
        min_occurs = 0
        max_occurs: Optional[int] = 0
        for item in self.iter_model():
            if isinstance(item, XsdGroup):
                return False
            elif item.min_occurs == 0 or item.is_restriction(other, False):
                min_occurs += item.min_occurs
                if max_occurs is not None:
                    if item.max_occurs is None:
                        max_occurs = None
                    else:
                        max_occurs += item.max_occurs
                continue
            return False
        if min_occurs < other.min_occurs:
            return False
        elif max_occurs is None:
            return other.max_occurs is None
        elif other.max_occurs is None:
            return True
        else:
            return max_occurs <= other.max_occurs