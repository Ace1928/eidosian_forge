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
def is_sequence_restriction(self, other: XsdGroup) -> bool:
    if not self.has_occurs_restriction(other):
        return False
    check_occurs = other.max_occurs != 0
    item_iterator = iter(self.iter_model())
    item = next(item_iterator, None)
    for other_item in other.iter_model():
        if item is not None and item.is_restriction(other_item, check_occurs):
            item = next(item_iterator, None)
        elif not other_item.is_emptiable():
            break
    else:
        if item is None:
            return True
    item_iterator = iter(self)
    item = next(item_iterator, None)
    for other_item in other.iter_model():
        if item is not None and item.is_restriction(other_item, check_occurs):
            item = next(item_iterator, None)
        elif not other_item.is_emptiable():
            break
    else:
        if item is None:
            return True
    other_items = other.iter_model()
    for other_item in other_items:
        if self.is_restriction(other_item, check_occurs):
            return all((x.is_emptiable() for x in other_items))
        elif not other_item.is_emptiable():
            return False
    else:
        return False