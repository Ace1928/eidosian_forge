from __future__ import annotations
import decimal
import struct
from typing import Any, Sequence, Tuple, Type, Union
@property
def bid(self) -> bytes:
    """The Binary Integer Decimal (BID) encoding of this instance."""
    return _PACK_64(self.__low) + _PACK_64(self.__high)