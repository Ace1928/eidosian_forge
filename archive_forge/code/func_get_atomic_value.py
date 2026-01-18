from decimal import Decimal
from typing import Dict, Optional, Union
from ..namespaces import XSD_NAMESPACE
from ..protocols import XsdTypeProtocol
from .atomic_types import xsd10_atomic_types, xsd11_atomic_types, \
from .untyped import UntypedAtomic
from .qname import AbstractQName, QName, Notation
from .numeric import Float10, Float, Integer, Int, NegativeInteger, \
from .string import NormalizedString, XsdToken, Name, NCName, NMToken, Id, \
from .uri import AnyURI
from .binary import AbstractBinary, Base64Binary, HexBinary
from .datetime import AbstractDateTime, DateTime10, DateTime, DateTimeStamp, \
from .proxies import BooleanProxy, DecimalProxy, DoubleProxy10, DoubleProxy, \
def get_atomic_value(xsd_type: Optional[XsdTypeProtocol]) -> AtomicValueType:
    """Gets an atomic value for an XSD type instance. Used for schema contexts."""
    if xsd_type is None:
        return UntypedAtomic('1')
    try:
        return ATOMIC_VALUES[xsd_type.name]
    except KeyError:
        try:
            return ATOMIC_VALUES[xsd_type.root_type.name]
        except KeyError:
            return UntypedAtomic('1')