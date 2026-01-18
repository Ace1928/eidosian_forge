from __future__ import annotations
import numpy as np
from . import imageglobals as imageglobals
from .batteryrunners import BatteryRunner
from .volumeutils import Recoder, endian_codes, native_code, pretty_mapping, swapped_code
@property
def endianness(self):
    """endian code of binary data

        The endianness code gives the current byte order
        interpretation of the binary data.

        Examples
        --------
        >>> wstr = WrapStruct()
        >>> code = wstr.endianness
        >>> code == native_code
        True

        Notes
        -----
        Endianness gives endian interpretation of binary data. It is
        read only because the only common use case is to set the
        endianness on initialization, or occasionally byteswapping the
        data - but this is done via the as_byteswapped method
        """
    if self._structarr.dtype.isnative:
        return native_code
    return swapped_code