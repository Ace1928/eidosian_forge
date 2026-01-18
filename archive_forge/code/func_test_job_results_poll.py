from unittest import mock
import warnings
import pytest
import cirq_ionq as ionq
@mock.patch('time.sleep', return_value=None)
def test_job_results_poll(mock_sleep):
    ready_job = {'id': 'my_id', 'status': 'ready'}
    completed_job = {'id': 'my_id', 'status': 'completed', 'qubits': '1', 'target': 'qpu', 'metadata': {'shots': 1000}}
    mock_client = mock.MagicMock()
    mock_client.get_job.side_effect = [ready_job, completed_job]
    mock_client.get_results.return_value = {'0': '0.6', '1': '0.4'}
    job = ionq.Job(mock_client, ready_job)
    results = job.results(polling_seconds=0)
    assert results == ionq.QPUResult({0: 600, 1: 400}, 1, measurement_dict={})
    mock_sleep.assert_called_once()