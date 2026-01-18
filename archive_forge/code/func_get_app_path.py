import glob
import os
import struct
from .lazy_import import lazy_import
import ctypes
from breezy import cmdline
from breezy.i18n import gettext
def get_app_path(appname):
    """Look up in Windows registry for full path to application executable.
    Typically, applications create subkey with their basename
    in HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\

    :param  appname:    name of application (if no filename extension
                        is specified, .exe used)
    :return:    full path to aplication executable from registry,
                or appname itself if nothing found.
    """
    import winreg
    basename = appname
    if not os.path.splitext(basename)[1]:
        basename = appname + '.exe'
    try:
        hkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\' + basename)
    except OSError:
        return appname
    try:
        try:
            path, type_id = winreg.QueryValueEx(hkey, '')
        except OSError:
            return appname
    finally:
        winreg.CloseKey(hkey)
    if type_id == REG_SZ:
        return path
    return appname