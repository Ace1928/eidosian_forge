from unittest import mock
import warnings
import pytest
import cirq_ionq as ionq
def test_job_sharpen_results():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {'0': '60', '1': '40'}
    job_dict = {'id': 'my_id', 'status': 'completed', 'qubits': '1', 'target': 'simulator', 'metadata': {'shots': '100'}}
    job = ionq.Job(mock_client, job_dict)
    results = job.results(sharpen=False)
    assert results == ionq.SimulatorResult({0: 60, 1: 40}, 1, {}, 100)