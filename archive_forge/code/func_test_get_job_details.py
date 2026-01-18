from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
@ddt.data({'extended': False, 'expected_fields': ['InstanceID']}, {'extended': True, 'expected_fields': ['InstanceID', 'DetailedStatus']})
@ddt.unpack
@mock.patch.object(jobutils.JobUtils, '_get_job_error_details')
def test_get_job_details(self, mock_get_job_err, expected_fields, extended):
    mock_job = mock.Mock()
    details = self.jobutils._get_job_details(mock_job, extended=extended)
    if extended:
        mock_get_job_err.assert_called_once_with(mock_job)
        self.assertEqual(details['RawErrors'], mock_get_job_err.return_value)
    for field in expected_fields:
        self.assertEqual(getattr(mock_job, field), details[field])