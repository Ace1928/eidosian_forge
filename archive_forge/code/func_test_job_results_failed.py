from unittest import mock
import warnings
import pytest
import cirq_ionq as ionq
def test_job_results_failed():
    job_dict = {'id': 'my_id', 'status': 'failed', 'failure': {'error': 'too many qubits'}}
    job = ionq.Job(None, job_dict)
    with pytest.raises(RuntimeError, match='too many qubits'):
        _ = job.results()
    assert job.status() == 'failed'