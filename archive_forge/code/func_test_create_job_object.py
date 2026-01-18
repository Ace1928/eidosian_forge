from unittest import mock
import ddt
from os_win.tests.unit import test_base
from os_win.utils import processutils
from os_win.utils.winapi import constants as w_const
def test_create_job_object(self):
    job_handle = self._procutils.create_job_object(mock.sentinel.name)
    self._mock_run.assert_called_once_with(self._mock_kernel32.CreateJobObjectW, None, self._ctypes.c_wchar_p(mock.sentinel.name), error_ret_vals=[None], kernel32_lib_func=True)
    self.assertEqual(self._mock_run.return_value, job_handle)