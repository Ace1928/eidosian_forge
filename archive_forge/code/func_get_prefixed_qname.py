import re
from collections import Counter
from decimal import Decimal
from typing import Any, Callable, Iterator, List, MutableMapping, \
from xml.etree.ElementTree import ParseError
from .exceptions import XMLSchemaValueError, XMLSchemaTypeError
from .names import XSI_SCHEMA_LOCATION, XSI_NONS_SCHEMA_LOCATION
from .aliases import ElementType, NamespacesType, AtomicValueType, NumericValueType
def get_prefixed_qname(qname: str, namespaces: Optional[MutableMapping[str, str]], use_empty: bool=True) -> str:
    """
    Get the prefixed form of a QName, using a namespace map.

    :param qname: an extended QName or a local name or a prefixed QName.
    :param namespaces: an optional mapping from prefixes to namespace URIs.
    :param use_empty: if `True` use the empty prefix for mapping.
    """
    if not namespaces or not qname or qname[0] != '{':
        return qname
    namespace = get_namespace(qname)
    prefixes = [x for x in namespaces if namespaces[x] == namespace]
    if not prefixes:
        return qname
    elif prefixes[0]:
        return f'{prefixes[0]}:{qname.split('}', 1)[1]}'
    elif len(prefixes) > 1:
        return f'{prefixes[1]}:{qname.split('}', 1)[1]}'
    elif use_empty:
        return qname.split('}', 1)[1]
    else:
        return qname