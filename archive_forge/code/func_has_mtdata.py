import os
import time
from ctypes import cdll, Structure, c_ulong, c_int, c_ushort, \
def has_mtdata(self):
    """Return True if the device has multitouch data.
        """
    if self._fd == -1:
        raise Exception('Device closed')
    return bool(self._device.caps.has_mtdata)