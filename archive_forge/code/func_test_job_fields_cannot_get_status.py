from unittest import mock
import warnings
import pytest
import cirq_ionq as ionq
def test_job_fields_cannot_get_status():
    job_dict = {'id': 'my_id', 'target': 'qpu', 'name': 'bacon', 'qubits': '5', 'status': 'running', 'metadata': {'shots': 1000}}
    mock_client = mock.MagicMock()
    mock_client.get_job.side_effect = ionq.IonQException('bad')
    job = ionq.Job(mock_client, job_dict)
    with pytest.raises(ionq.IonQException, match='bad'):
        _ = job.target()
    with pytest.raises(ionq.IonQException, match='bad'):
        _ = job.name()
    with pytest.raises(ionq.IonQException, match='bad'):
        _ = job.num_qubits()
    with pytest.raises(ionq.IonQException, match='bad'):
        _ = job.repetitions()