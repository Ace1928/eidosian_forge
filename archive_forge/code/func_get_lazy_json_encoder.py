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
def get_lazy_json_encoder(errors: List[XMLSchemaValidationError]) -> Type[json.JSONEncoder]:

    class JSONLazyEncoder(json.JSONEncoder):

        def default(self, obj: Any) -> Any:
            if isinstance(obj, Iterator):
                for result in obj:
                    if isinstance(result, XMLSchemaValidationError):
                        errors.append(result)
                    else:
                        return result
                return None
            return json.JSONEncoder.default(self, obj)
    return JSONLazyEncoder