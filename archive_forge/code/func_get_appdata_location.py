import glob
import os
import struct
from .lazy_import import lazy_import
import ctypes
from breezy import cmdline
from breezy.i18n import gettext
def get_appdata_location():
    """Return Application Data location.
    Return None if we cannot obtain location.

    Windows defines two 'Application Data' folders per user - a 'roaming'
    one that moves with the user as they logon to different machines, and
    a 'local' one that stays local to the machine.  This returns the 'roaming'
    directory, and thus is suitable for storing user-preferences, etc.
    """
    appdata = _get_sh_special_folder_path(CSIDL_APPDATA)
    if appdata:
        return appdata
    return os.environ.get('APPDATA')