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
def is_all_restriction(self, other: XsdGroup) -> bool:
    restriction_items = [x for x in self.iter_model()]
    base_items = [x for x in other.iter_model()]
    wildcards: List[XsdAnyElement] = []
    for w1 in base_items:
        if isinstance(w1, XsdAnyElement):
            for w2 in wildcards:
                if w1.process_contents == w2.process_contents and w1.occurs == w2.occurs:
                    w2.union(w1)
                    w2.extended = True
                    break
            else:
                wildcards.append(_copy(w1))
    base_items.extend((w for w in wildcards if hasattr(w, 'extended')))
    if self.model != 'choice':
        restriction_wildcards = [e for e in restriction_items if isinstance(e, XsdAnyElement)]
        for other_item in base_items:
            min_occurs, max_occurs = (0, other_item.max_occurs)
            for k in range(len(restriction_items) - 1, -1, -1):
                item = restriction_items[k]
                if item.is_restriction(other_item, check_occurs=False):
                    if max_occurs is None:
                        min_occurs += item.min_occurs
                    elif item.max_occurs is None or max_occurs < item.max_occurs or min_occurs + item.min_occurs > max_occurs:
                        continue
                    else:
                        min_occurs += item.min_occurs
                        max_occurs -= item.max_occurs
                    restriction_items.remove(item)
                    if not min_occurs or max_occurs == 0:
                        break
            else:
                if self.model == 'all' and restriction_wildcards:
                    if not isinstance(other_item, XsdGroup) and other_item.type and (other_item.type.name != XSD_ANY_TYPE):
                        for w in restriction_wildcards:
                            if w.is_matching(other_item.name, self.target_namespace):
                                return False
            if min_occurs < other_item.min_occurs:
                break
        else:
            if not restriction_items:
                return True
        return False
    not_emptiable_items = {x for x in base_items if x.min_occurs}
    for other_item in base_items:
        min_occurs, max_occurs = (0, other_item.max_occurs)
        for k in range(len(restriction_items) - 1, -1, -1):
            item = restriction_items[k]
            if item.is_restriction(other_item, check_occurs=False):
                if max_occurs is None:
                    min_occurs += item.min_occurs
                elif item.max_occurs is None or max_occurs < item.max_occurs or min_occurs + item.min_occurs > max_occurs:
                    continue
                else:
                    min_occurs += item.min_occurs
                    max_occurs -= item.max_occurs
                if not_emptiable_items:
                    if len(not_emptiable_items) > 1:
                        continue
                    if other_item not in not_emptiable_items:
                        continue
                restriction_items.remove(item)
                if not min_occurs or max_occurs == 0:
                    break
        if min_occurs < other_item.min_occurs:
            break
    else:
        if not restriction_items:
            return True
    if any((not isinstance(x, XsdGroup) for x in restriction_items)):
        return False
    for group in restriction_items:
        if not group.is_restriction(other):
            return False
        for item in not_emptiable_items:
            for e in group:
                if e.name == item.name:
                    break
            else:
                return False
    else:
        return True