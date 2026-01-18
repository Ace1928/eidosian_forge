from unittest import mock
import warnings
import pytest
import cirq_ionq as ionq
def test_job_fields_update_status():
    job_dict = {'id': 'my_id', 'target': 'qpu', 'name': 'bacon', 'qubits': '5', 'status': 'running', 'metadata': {'shots': 1000}}
    mock_client = mock.MagicMock()
    mock_client.get_job.return_value = job_dict
    job = ionq.Job(mock_client, job_dict)
    assert job.job_id() == 'my_id'
    assert job.target() == 'qpu'
    assert job.name() == 'bacon'
    assert job.num_qubits() == 5
    assert job.repetitions() == 1000