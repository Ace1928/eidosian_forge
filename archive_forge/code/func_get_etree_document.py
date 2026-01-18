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
def get_etree_document(self) -> Any:
    """
        The resource as ElementTree XML document. If the resource is lazy
        raises a resource error.
        """
    if is_etree_document(self._source):
        return self._source
    elif self._lazy:
        raise XMLResourceError('cannot create an ElementTree instance from a lazy XML resource')
    elif hasattr(self._root, 'nsmap'):
        return self._root.getroottree()
    else:
        return ElementTree.ElementTree(self._root)