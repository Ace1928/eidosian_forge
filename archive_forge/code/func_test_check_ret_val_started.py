from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
@mock.patch.object(jobutils.JobUtils, '_wait_for_job')
def test_check_ret_val_started(self, mock_wait_for_job):
    self.jobutils.check_ret_val(constants.WMI_JOB_STATUS_STARTED, mock.sentinel.job_path)
    mock_wait_for_job.assert_called_once_with(mock.sentinel.job_path)