from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
@mock.patch.object(jobutils.JobUtils, '_wait_for_job')
def test_check_ret_val_ok(self, mock_wait_for_job):
    self.jobutils.check_ret_val(self._FAKE_RET_VAL, mock.sentinel.job_path)
    self.assertFalse(mock_wait_for_job.called)