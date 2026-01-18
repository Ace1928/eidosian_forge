from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
def test_get_job_error_details_exception(self):
    mock_job = mock.Mock()
    mock_job.GetErrorEx.side_effect = Exception
    self.assertIsNone(self.jobutils._get_job_error_details(mock_job))