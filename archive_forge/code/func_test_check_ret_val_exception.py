from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
def test_check_ret_val_exception(self):
    self.assertRaises(exceptions.WMIJobFailed, self.jobutils.check_ret_val, mock.sentinel.ret_val_bad, mock.sentinel.job_path)