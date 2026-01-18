from abc import ABCMeta
import os
import logging
import threading
import warnings
import re
import sys
from copy import copy as _copy
from operator import attrgetter
from typing import cast, Callable, ItemsView, List, Optional, Dict, Any, \
from xml.etree.ElementTree import Element, ParseError
from elementpath import XPathToken, SchemaElementNode, build_schema_node_tree
from ..exceptions import XMLSchemaTypeError, XMLSchemaKeyError, XMLSchemaRuntimeError, \
from ..names import VC_MIN_VERSION, VC_MAX_VERSION, VC_TYPE_AVAILABLE, \
from ..aliases import ElementType, XMLSourceType, NamespacesType, LocationsType, \
from ..translation import gettext as _
from ..helpers import prune_etree, get_namespace, get_qname, is_defuse_error
from ..namespaces import NamespaceResourcesMap, NamespaceView
from ..resources import is_local_url, is_remote_url, url_path_is_file, \
from ..converters import XMLSchemaConverter
from ..xpath import XsdSchemaProtocol, XMLSchemaProxy, ElementPathMixin
from .. import dataobjects
from .exceptions import XMLSchemaParseError, XMLSchemaValidationError, XMLSchemaEncodeError, \
from .helpers import get_xsd_derivation_attribute
from .xsdbase import check_validation_mode, XsdValidator, XsdComponent, XsdAnnotation
from .notations import XsdNotation
from .identities import XsdIdentity, XsdKey, XsdKeyref, XsdUnique, \
from .facets import XSD_10_FACETS, XSD_11_FACETS
from .simple_types import XsdSimpleType, XsdList, XsdUnion, XsdAtomicRestriction, \
from .attributes import XsdAttribute, XsdAttributeGroup, Xsd11Attribute
from .complex_types import XsdComplexType, Xsd11ComplexType
from .groups import XsdGroup, Xsd11Group
from .elements import XsdElement, Xsd11Element
from .wildcards import XsdAnyElement, XsdAnyAttribute, Xsd11AnyElement, \
from .global_maps import XsdGlobals
def create_any_content_group(self, parent: Union[XsdComplexType, XsdGroup], any_element: Optional[XsdAnyElement]=None) -> XsdGroup:
    """
        Creates a model group related to schema instance that accepts any content.

        :param parent: the parent component to set for the content group.
        :param any_element: an optional any element to use for the content group.         When provided it's copied, linked to the group and the minOccurs/maxOccurs         are set to 0 and 'unbounded'.
        """
    group: XsdGroup = self.xsd_group_class(SEQUENCE_ELEMENT, self, parent)
    if isinstance(any_element, XsdAnyElement):
        particle = _copy(any_element)
        particle.min_occurs = 0
        particle.max_occurs = None
        particle.parent = group
        group.append(particle)
    else:
        group.append(self.xsd_any_class(ANY_ELEMENT, self, group))
    return group