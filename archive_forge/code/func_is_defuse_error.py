import re
from collections import Counter
from decimal import Decimal
from typing import Any, Callable, Iterator, List, MutableMapping, \
from xml.etree.ElementTree import ParseError
from .exceptions import XMLSchemaValueError, XMLSchemaTypeError
from .names import XSI_SCHEMA_LOCATION, XSI_NONS_SCHEMA_LOCATION
from .aliases import ElementType, NamespacesType, AtomicValueType, NumericValueType
def is_defuse_error(err: Exception) -> bool:
    """
    Returns `True` if the error is related to defuse of XML data in the DTD
    of the source (forbid entities or external references), `False` otherwise.
    """
    if not isinstance(err, ParseError):
        return False
    msg = str(err)
    return 'Entities are forbidden' in msg or 'Unparsed entities are forbidden' in msg or 'External references are forbidden' in msg