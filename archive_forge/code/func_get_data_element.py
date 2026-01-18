import re
from abc import ABCMeta
from copy import copy
from itertools import count
from typing import TYPE_CHECKING, cast, overload, Any, Dict, List, Iterator, \
from elementpath import XPathContext, XPath2Parser, build_node_tree, protocols
from elementpath.etree import etree_tostring
from .exceptions import XMLSchemaAttributeError, XMLSchemaTypeError, XMLSchemaValueError
from .aliases import ElementType, XMLSourceType, NamespacesType, BaseXsdType, DecodeType
from .helpers import get_namespace, get_prefixed_qname, local_name, raw_xml_encode
from .converters import ElementData, XMLSchemaConverter
from .resources import XMLResource
from . import validators
def get_data_element(self, data: ElementData, xsd_element: 'XsdElement', xsd_type: Optional[BaseXsdType]=None) -> DataElement:
    cls = xsd_element.get_binding(self.data_element_class)
    return cls(tag=data.tag, value=data.text, nsmap=self.namespaces, xsd_type=xsd_type)