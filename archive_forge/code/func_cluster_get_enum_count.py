import contextlib
import ctypes
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def cluster_get_enum_count(self, enum_handle):
    return self._run_and_check_output(clusapi.ClusterGetEnumCountEx, enum_handle, error_on_nonzero_ret_val=False, ret_val_is_err_code=False)