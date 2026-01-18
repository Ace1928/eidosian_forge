import glob
import os
import struct
from .lazy_import import lazy_import
import ctypes
from breezy import cmdline
from breezy.i18n import gettext
def get_fs_type(drive):
    """Return file system type for a drive on the system.

    Args:
      drive: Unicode string with drive including trailing backslash (e.g.
         "C:\\")
    Returns:
      Windows filesystem type name (e.g. "FAT32", "NTFS") or None
      if the drive can not be found
    """
    MAX_FS_TYPE_LENGTH = 16
    kernel32 = ctypes.windll.kernel32
    GetVolumeInformation = kernel32.GetVolumeInformationW
    fs_type = ctypes.create_unicode_buffer(MAX_FS_TYPE_LENGTH + 1)
    if GetVolumeInformation(drive, None, 0, None, None, None, fs_type, MAX_FS_TYPE_LENGTH):
        return fs_type.value
    return None