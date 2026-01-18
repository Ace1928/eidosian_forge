from unittest import mock
import warnings
import pytest
import cirq_ionq as ionq
def test_job_status_refresh():
    for status in ionq.Job.NON_TERMINAL_STATES:
        mock_client = mock.MagicMock()
        mock_client.get_job.return_value = {'id': 'my_id', 'status': 'completed'}
        job = ionq.Job(mock_client, {'id': 'my_id', 'status': status})
        assert job.status() == 'completed'
        mock_client.get_job.assert_called_with('my_id')
    for status in ionq.Job.TERMINAL_STATES:
        mock_client = mock.MagicMock()
        job = ionq.Job(mock_client, {'id': 'my_id', 'status': status})
        assert job.status() == status
        mock_client.get_job.assert_not_called()