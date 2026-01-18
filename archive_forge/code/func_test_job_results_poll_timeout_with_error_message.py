from unittest import mock
import warnings
import pytest
import cirq_ionq as ionq
@mock.patch('time.sleep', return_value=None)
def test_job_results_poll_timeout_with_error_message(mock_sleep):
    ready_job = {'id': 'my_id', 'status': 'failure', 'failure': {'error': 'too many qubits'}}
    mock_client = mock.MagicMock()
    mock_client.get_job.return_value = ready_job
    job = ionq.Job(mock_client, ready_job)
    with pytest.raises(RuntimeError, match='too many qubits'):
        _ = job.results(timeout_seconds=1, polling_seconds=0.1)
    assert mock_sleep.call_count == 11