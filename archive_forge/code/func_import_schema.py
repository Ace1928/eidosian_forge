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
def import_schema(self, namespace: str, location: str, base_url: Optional[str]=None, force: bool=False, build: bool=False) -> Optional[SchemaType]:
    """
        Imports a schema for an external namespace, from a specific URL.

        :param namespace: is the URI of the external namespace.
        :param location: is the URL of the schema.
        :param base_url: is an optional base URL for fetching the schema resource.
        :param force: if set to `True` imports the schema also if the namespace is already imported.
        :param build: defines when to build the imported schema, the default is to not build.
        :return: the imported :class:`XMLSchema` instance.
        """
    if location == self.url:
        return self
    if not force:
        if self.imports.get(namespace) is not None:
            return self.imports[namespace]
        elif namespace in self.maps.namespaces:
            self.imports[namespace] = self.maps.namespaces[namespace][0]
            return self.imports[namespace]
    schema: SchemaType
    schema_url = fetch_resource(location, base_url)
    imported_ns = self.imports.get(namespace)
    if imported_ns is not None and imported_ns.url == schema_url:
        return imported_ns
    elif namespace in self.maps.namespaces:
        for schema in self.maps.namespaces[namespace]:
            if schema_url == schema.url:
                self.imports[namespace] = schema
                return schema
    schema = type(self)(source=schema_url, validation=self.validation, global_maps=self.maps, converter=self.converter, locations=[x for x in self._locations if x[0] != namespace], base_url=self.base_url, allow=self.allow, defuse=self.defuse, timeout=self.timeout, build=build)
    if schema.target_namespace != namespace:
        msg = _('imported schema {0!r} has an unmatched namespace {1!r}')
        raise XMLSchemaValueError(msg.format(location, namespace))
    self.imports[namespace] = schema
    return schema