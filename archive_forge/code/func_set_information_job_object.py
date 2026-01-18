import ctypes
from oslo_log import log as logging
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import kernel32 as kernel32_struct
def set_information_job_object(self, job_handle, job_object_info_class, job_object_info):
    self._run_and_check_output(kernel32.SetInformationJobObject, job_handle, job_object_info_class, ctypes.byref(job_object_info), ctypes.sizeof(job_object_info))