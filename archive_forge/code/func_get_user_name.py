import glob
import os
import struct
from .lazy_import import lazy_import
import ctypes
from breezy import cmdline
from breezy.i18n import gettext
def get_user_name():
    """Return user name as login name.
    If name cannot be obtained return None.
    """
    try:
        advapi32 = ctypes.windll.advapi32
        GetUserName = getattr(advapi32, 'GetUserNameW')
    except AttributeError:
        pass
    else:
        buf = ctypes.create_unicode_buffer(UNLEN + 1)
        n = ctypes.c_int(UNLEN + 1)
        if GetUserName(buf, ctypes.byref(n)):
            return buf.value
    return os.environ.get('USERNAME')