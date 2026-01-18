from unittest import mock
import warnings
import pytest
import cirq_ionq as ionq
def test_job_str():
    job = ionq.Job(None, {'id': 'my_id'})
    assert str(job) == 'cirq_ionq.Job(job_id=my_id)'