import re
from typing import TYPE_CHECKING, cast, Any, Dict, Generic, List, Iterator, Optional, \
from xml.etree import ElementTree
from elementpath import select
from elementpath.etree import is_etree_element, etree_tostring
from ..exceptions import XMLSchemaValueError, XMLSchemaTypeError
from ..names import XSD_ANNOTATION, XSD_APPINFO, XSD_DOCUMENTATION, \
from ..aliases import ElementType, NamespacesType, SchemaType, BaseXsdType, \
from ..translation import gettext as _
from ..helpers import get_qname, local_name, get_prefixed_qname
from ..resources import XMLResource
from .exceptions import XMLSchemaParseError, XMLSchemaValidationError
def get_matching_item(self, mapping: MutableMapping[str, Any], ns_prefix: str='xmlns', match_local_name: bool=False) -> Optional[Any]:
    """
        If a key is matching component name, returns its value, otherwise returns `None`.
        """
    if self.name is None:
        return None
    elif not self.target_namespace:
        return mapping.get(self.name)
    elif self.qualified_name in mapping:
        return mapping[cast(str, self.qualified_name)]
    elif self.prefixed_name in mapping:
        return mapping[cast(str, self.prefixed_name)]
    target_namespace = self.target_namespace
    suffix = f':{self.local_name}'
    for k in filter(lambda x: x.endswith(suffix), mapping):
        prefix = k.split(':')[0]
        if self.namespaces.get(prefix) == target_namespace:
            return mapping[k]
        ns_declaration = '{}:{}'.format(ns_prefix, prefix)
        try:
            if mapping[k][ns_declaration] == target_namespace:
                return mapping[k]
        except (KeyError, TypeError):
            pass
    else:
        if match_local_name:
            return mapping.get(self.local_name)
        return None