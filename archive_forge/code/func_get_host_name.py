import glob
import os
import struct
from .lazy_import import lazy_import
import ctypes
from breezy import cmdline
from breezy.i18n import gettext
def get_host_name():
    """Return host machine name.
    If name cannot be obtained return None.

    :return: A unicode string representing the host name.
    """
    buf = ctypes.create_unicode_buffer(MAX_COMPUTERNAME_LENGTH + 1)
    n = ctypes.c_int(MAX_COMPUTERNAME_LENGTH + 1)
    GetComputerNameEx = getattr(ctypes.windll.kernel32, 'GetComputerNameExW', None)
    if GetComputerNameEx is not None and GetComputerNameEx(_WIN32_ComputerNameDnsHostname, buf, ctypes.byref(n)):
        return buf.value
    return os.environ.get('COMPUTERNAME')