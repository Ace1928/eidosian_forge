import contextlib
import ctypes
import os
import shutil
import tempfile
from oslo_log import log as logging
from oslo_utils import fileutils
from os_win._i18n import _
from os_win import _utils
import os_win.conf
from os_win import exceptions
from os_win.utils import _acl_utils
from os_win.utils.io import ioutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import advapi32 as advapi32_def
from os_win.utils.winapi.libs import kernel32 as kernel32_def
from os_win.utils.winapi import wintypes
def get_file_id(self, path):
    """Return a dict containing the file id and volume id."""
    handle = None
    info = kernel32_def.FILE_ID_INFO()
    try:
        handle = self._io_utils.open(path, desired_access=0, share_mode=w_const.FILE_SHARE_READ | w_const.FILE_SHARE_WRITE | w_const.FILE_SHARE_DELETE, creation_disposition=w_const.OPEN_EXISTING)
        self._win32_utils.run_and_check_output(kernel32.GetFileInformationByHandleEx, handle, w_const.FileIdInfo, ctypes.byref(info), ctypes.sizeof(info), kernel32_lib_func=True)
    finally:
        if handle:
            self._io_utils.close_handle(handle)
    return dict(volume_serial_number=info.VolumeSerialNumber, file_id=bytearray(info.FileId.Identifier))