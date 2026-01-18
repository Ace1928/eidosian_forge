import re
from collections import Counter
from decimal import Decimal
from typing import Any, Callable, Iterator, List, MutableMapping, \
from xml.etree.ElementTree import ParseError
from .exceptions import XMLSchemaValueError, XMLSchemaTypeError
from .names import XSI_SCHEMA_LOCATION, XSI_NONS_SCHEMA_LOCATION
from .aliases import ElementType, NamespacesType, AtomicValueType, NumericValueType
def raw_xml_encode(value: Union[None, AtomicValueType, List[AtomicValueType], Tuple[AtomicValueType, ...]]) -> Optional[str]:
    """Encodes a simple value to XML."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    elif isinstance(value, (list, tuple)):
        return ' '.join((str(e) for e in value))
    else:
        return str(value) if value is not None else None