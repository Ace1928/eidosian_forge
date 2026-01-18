from unittest import mock
import warnings
import pytest
import cirq_ionq as ionq
def test_job_fields_unsuccessful():
    job_dict = {'id': 'my_id', 'target': 'qpu', 'name': 'bacon', 'qubits': '5', 'status': 'deleted', 'metadata': {'shots': 1000}}
    job = ionq.Job(None, job_dict)
    with pytest.raises(ionq.IonQUnsuccessfulJobException, match='deleted'):
        _ = job.target()
    with pytest.raises(ionq.IonQUnsuccessfulJobException, match='deleted'):
        _ = job.name()
    with pytest.raises(ionq.IonQUnsuccessfulJobException, match='deleted'):
        _ = job.num_qubits()
    with pytest.raises(ionq.IonQUnsuccessfulJobException, match='deleted'):
        _ = job.repetitions()