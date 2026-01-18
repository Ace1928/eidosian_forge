from unittest import mock
import ddt
from os_win.tests.unit import test_base
from os_win.utils import processutils
from os_win.utils.winapi import constants as w_const
def test_assign_process_to_job_object(self):
    self._procutils.assign_process_to_job_object(mock.sentinel.job_handle, mock.sentinel.process_handle)
    self._mock_run.assert_called_once_with(self._mock_kernel32.AssignProcessToJobObject, mock.sentinel.job_handle, mock.sentinel.process_handle, kernel32_lib_func=True)