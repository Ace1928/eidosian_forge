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
def check_dynamic_context(self, elem: ElementType, xsd_element: SchemaElementType, model_element: SchemaElementType, namespaces: NamespacesType) -> None:
    if model_element is not xsd_element and isinstance(model_element, XsdElement):
        if 'substitution' in model_element.block or (xsd_element.type and xsd_element.type.is_blocked(model_element)):
            reason = _('substitution of %r is blocked') % model_element
            raise XMLSchemaValidationError(model_element, elem, reason)
    alternatives: Union[Tuple[()], List[XsdAlternative]] = []
    if isinstance(xsd_element, XsdAnyElement):
        if xsd_element.process_contents == 'skip':
            return
        try:
            xsd_element = self.maps.lookup_element(elem.tag)
        except LookupError:
            if self.schema.meta_schema is None:
                return
            try:
                type_name = elem.attrib[XSI_TYPE].strip()
            except KeyError:
                return
            else:
                xsd_type = self.maps.get_instance_type(type_name, self.any_type, namespaces)
        else:
            alternatives = xsd_element.alternatives
            try:
                type_name = elem.attrib[XSI_TYPE].strip()
            except KeyError:
                xsd_type = xsd_element.type
            else:
                xsd_type = self.maps.get_instance_type(type_name, xsd_element.type, namespaces)
    else:
        if XSI_TYPE not in elem.attrib or self.schema.meta_schema is None:
            xsd_type = xsd_element.type
        else:
            alternatives = xsd_element.alternatives
            try:
                type_name = elem.attrib[XSI_TYPE].strip()
            except KeyError:
                xsd_type = xsd_element.type
            else:
                xsd_type = self.maps.get_instance_type(type_name, xsd_element.type, namespaces)
        if model_element is not xsd_element and isinstance(model_element, XsdElement) and model_element.block:
            for derivation in model_element.block.split():
                if xsd_type is not model_element.type and xsd_type.is_derived(model_element.type, derivation):
                    reason = _('usage of {0!r} with type {1} is blocked by head element').format(xsd_element, derivation)
                    raise XMLSchemaValidationError(self, elem, reason)
        if XSI_TYPE not in elem.attrib or self.schema.meta_schema is None:
            return
    group = self.restriction if self.restriction is not None else self
    for e in group.iter_elements():
        if not isinstance(e, XsdElement):
            continue
        elif e.name == elem.tag:
            other = e
        else:
            for other in e.iter_substitutes():
                if other.name == elem.tag:
                    break
            else:
                continue
        if len(other.alternatives) != len(alternatives) or not xsd_type.is_dynamic_consistent(other.type):
            reason = _('{0!r} that matches {1!r} is not consistent with local declaration {2!r}').format(elem, xsd_element, other)
            raise XMLSchemaValidationError(self, reason)
        if not all((any((a == x for x in alternatives)) for a in other.alternatives)) or not all((any((a == x for x in other.alternatives)) for a in alternatives)):
            msg = _('Maybe a not equivalent type table between elements {0!r} and {1!r}.').format(self, xsd_element)
            warnings.warn(msg, XMLSchemaTypeTableWarning, stacklevel=3)