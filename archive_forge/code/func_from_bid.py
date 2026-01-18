from __future__ import annotations
import decimal
import struct
from typing import Any, Sequence, Tuple, Type, Union
@classmethod
def from_bid(cls: Type[Decimal128], value: bytes) -> Decimal128:
    """Create an instance of :class:`Decimal128` from Binary Integer
        Decimal string.

        :Parameters:
          - `value`: 16 byte string (128-bit IEEE 754-2008 decimal floating
            point in Binary Integer Decimal (BID) format).
        """
    if not isinstance(value, bytes):
        raise TypeError('value must be an instance of bytes')
    if len(value) != 16:
        raise ValueError('value must be exactly 16 bytes')
    return cls((_UNPACK_64(value[8:])[0], _UNPACK_64(value[:8])[0]))