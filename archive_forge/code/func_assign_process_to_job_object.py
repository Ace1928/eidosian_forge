import ctypes
from oslo_log import log as logging
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import kernel32 as kernel32_struct
def assign_process_to_job_object(self, job_handle, process_handle):
    self._run_and_check_output(kernel32.AssignProcessToJobObject, job_handle, process_handle)