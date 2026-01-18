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
def resolve_qname(self, qname: str, namespace_imported: bool=True) -> str:
    """
        QName resolution for a schema instance.

        :param qname: a string in xs:QName format.
        :param namespace_imported: if this argument is `True` raises an         `XMLSchemaNamespaceError` if the namespace of the QName is not the         *targetNamespace* and the namespace is not imported by the schema.
        :returns: an expanded QName in the format "{*namespace-URI*}*local-name*".
        :raises: `XMLSchemaValueError` for an invalid xs:QName is found,         `XMLSchemaKeyError` if the namespace prefix is not declared in the         schema instance.
        """
    qname = qname.strip()
    if not qname or ' ' in qname or '\t' in qname or ('\n' in qname):
        msg = _('{!r} is not a valid value for xs:QName')
        raise XMLSchemaValueError(msg.format(qname))
    if qname[0] == '{':
        try:
            namespace, local_name = qname[1:].split('}')
        except ValueError:
            msg = _('{!r} is not a valid value for xs:QName')
            raise XMLSchemaValueError(msg.format(qname))
    elif ':' in qname:
        try:
            prefix, local_name = qname.split(':')
        except ValueError:
            msg = _('{!r} is not a valid value for xs:QName')
            raise XMLSchemaValueError(msg.format(qname))
        else:
            try:
                namespace = self.namespaces[prefix]
            except KeyError:
                msg = _('prefix {!r} not found in namespace map')
                raise XMLSchemaKeyError(msg.format(prefix))
    else:
        namespace, local_name = (self.namespaces.get('', ''), qname)
    if not namespace:
        if namespace_imported and self.target_namespace and ('' not in self.imports):
            msg = _("the QName {!r} is mapped to no namespace, but this requires that there is an xs:import statement in the schema without the 'namespace' attribute.")
            raise XMLSchemaNamespaceError(msg.format(qname))
        return local_name
    elif namespace_imported and self.meta_schema is not None and (namespace != self.target_namespace) and (namespace not in {XSD_NAMESPACE, XSI_NAMESPACE}) and (namespace not in self.imports):
        msg = _('the QName {0!r} is mapped to the namespace {1!r}, but this namespace has not an xs:import statement in the schema.')
        raise XMLSchemaNamespaceError(msg.format(qname, namespace))
    return f'{{{namespace}}}{local_name}'