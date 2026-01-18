import glob
import os
import struct
from .lazy_import import lazy_import
import ctypes
from breezy import cmdline
from breezy.i18n import gettext
def get_home_location():
    """Return user's home location.
    Assume on win32 it's the <My Documents> folder.
    If location cannot be obtained return system drive root,
    i.e. C:    """
    home = _get_sh_special_folder_path(CSIDL_PERSONAL)
    if home:
        return home
    home = os.environ.get('HOME')
    if home is not None:
        return home
    homepath = os.environ.get('HOMEPATH')
    if homepath is not None:
        return os.path.join(os.environ.get('HOMEDIR', ''), home)
    windir = os.environ.get('WINDIR')
    if windir:
        return os.path.splitdrive(windir)[0] + '/'
    return 'C:/'