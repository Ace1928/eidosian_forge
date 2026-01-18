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
def is_choice_restriction(self, other: XsdGroup) -> bool:
    restriction_items = [x for x in self.iter_model()]
    has_not_empty_item = any((e.max_occurs != 0 for e in restriction_items))
    check_occurs = other.max_occurs != 0
    max_occurs: Optional[int] = 0
    other_max_occurs: Optional[int] = 0
    for other_item in other.iter_model():
        for item in restriction_items:
            if other_item is item or item.is_restriction(other_item, check_occurs):
                if max_occurs is not None:
                    effective_max_occurs = item.effective_max_occurs
                    if effective_max_occurs is None:
                        max_occurs = None
                    elif self.model == 'choice':
                        max_occurs = max(max_occurs, effective_max_occurs)
                    else:
                        max_occurs += effective_max_occurs
                if other_max_occurs is not None:
                    effective_max_occurs = other_item.effective_max_occurs
                    if effective_max_occurs is None:
                        other_max_occurs = None
                    else:
                        other_max_occurs = max(other_max_occurs, effective_max_occurs)
                break
            elif item.max_occurs != 0:
                continue
            elif not other_item.is_matching(item.name):
                continue
            elif has_not_empty_item:
                break
            else:
                return False
        else:
            continue
        restriction_items.remove(item)
    if restriction_items:
        return False
    elif other_max_occurs is None:
        if other.max_occurs != 0:
            return True
        other_max_occurs = 0
    elif other.max_occurs is None:
        if other_max_occurs != 0:
            return True
        other_max_occurs = 0
    else:
        other_max_occurs *= other.max_occurs
    if max_occurs is None:
        return self.max_occurs == 0
    elif self.max_occurs is None:
        return max_occurs == 0
    else:
        return other_max_occurs >= max_occurs * self.max_occurs