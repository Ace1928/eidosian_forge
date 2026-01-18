import datetime
import struct
from six.moves import winreg
from six import text_type
from ._common import tzrangebase
def load_name(self, offset):
    """
        Load a timezone name from a DLL offset (integer).

        >>> from dateutil.tzwin import tzres
        >>> tzr = tzres()
        >>> print(tzr.load_name(112))
        'Eastern Standard Time'

        :param offset:
            A positive integer value referring to a string from the tzres dll.

        .. note::

            Offsets found in the registry are generally of the form
            ``@tzres.dll,-114``. The offset in this case is 114, not -114.

        """
    resource = self.p_wchar()
    lpBuffer = ctypes.cast(ctypes.byref(resource), wintypes.LPWSTR)
    nchar = self.LoadStringW(self._tzres._handle, offset, lpBuffer, 0)
    return resource[:nchar]