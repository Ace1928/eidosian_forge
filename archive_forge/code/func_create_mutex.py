import ctypes
from oslo_log import log as logging
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import kernel32 as kernel32_struct
def create_mutex(self, name=None, initial_owner=False, security_attributes=None):
    sec_attr_ref = ctypes.byref(security_attributes) if security_attributes else None
    return self._run_and_check_output(kernel32.CreateMutexW, sec_attr_ref, initial_owner, name)