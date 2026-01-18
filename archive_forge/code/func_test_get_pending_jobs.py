from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
def test_get_pending_jobs(self):
    mock_killed_job = mock.Mock(JobState=constants.JOB_STATE_KILLED)
    mock_running_job = mock.Mock(JobState=constants.WMI_JOB_STATE_RUNNING)
    mock_error_st_job = mock.Mock(JobState=constants.JOB_STATE_EXCEPTION)
    mappings = [mock.Mock(AffectingElement=None), mock.Mock(AffectingElement=mock_killed_job), mock.Mock(AffectingElement=mock_running_job), mock.Mock(AffectingElement=mock_error_st_job)]
    self.jobutils._conn.Msvm_AffectedJobElement.return_value = mappings
    mock_affected_element = mock.Mock()
    expected_pending_jobs = [mock_running_job]
    pending_jobs = self.jobutils._get_pending_jobs_affecting_element(mock_affected_element)
    self.assertEqual(expected_pending_jobs, pending_jobs)
    self.jobutils._conn.Msvm_AffectedJobElement.assert_called_once_with(AffectedElement=mock_affected_element.path_.return_value)