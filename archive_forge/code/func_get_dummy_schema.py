import json
import copy
from io import IOBase, TextIOBase
from typing import Any, Dict, List, Optional, Type, Union, Tuple, \
from elementpath.etree import ElementTree, etree_tostring
from .exceptions import XMLSchemaTypeError, XMLSchemaValueError, XMLResourceError
from .names import XSD_NAMESPACE, XSI_TYPE, XSD_SCHEMA
from .aliases import ElementType, XMLSourceType, NamespacesType, LocationsType, \
from .helpers import get_extended_qname, is_etree_document
from .resources import fetch_schema_locations, XMLResource
from .validators import XMLSchema10, XMLSchemaBase, XMLSchemaValidationError
def get_dummy_schema(tag: str, cls: Type[XMLSchemaBase]) -> XMLSchemaBase:
    if tag.startswith('{'):
        namespace, name = tag[1:].split('}')
    else:
        namespace, name = ('', tag)
    if namespace:
        return cls('<xs:schema xmlns:xs="{0}" targetNamespace="{1}">\n    <xs:element name="{2}"/>\n</xs:schema>'.format(XSD_NAMESPACE, namespace, name))
    else:
        return cls('<xs:schema xmlns:xs="{0}">\n    <xs:element name="{1}"/>\n</xs:schema>'.format(XSD_NAMESPACE, name))