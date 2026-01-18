from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
def test_get_job_error_details(self):
    mock_job = mock.Mock()
    error_details = self.jobutils._get_job_error_details(mock_job)
    mock_job.GetErrorEx.assert_called_once_with()
    self.assertEqual(mock_job.GetErrorEx.return_value, error_details)