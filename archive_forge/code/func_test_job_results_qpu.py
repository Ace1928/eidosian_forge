from unittest import mock
import warnings
import pytest
import cirq_ionq as ionq
def test_job_results_qpu():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {'0': '0.6', '2': '0.4'}
    job_dict = {'id': 'my_id', 'status': 'completed', 'qubits': '2', 'target': 'qpu', 'metadata': {'shots': 1000, 'measurement0': f'a{chr(31)}0,1'}, 'warning': {'messages': ['foo', 'bar']}}
    job = ionq.Job(mock_client, job_dict)
    with warnings.catch_warnings(record=True) as w:
        results = job.results()
        assert len(w) == 2
        assert 'foo' in str(w[0].message)
        assert 'bar' in str(w[1].message)
    expected = ionq.QPUResult({0: 600, 1: 400}, 2, {'a': [0, 1]})
    assert results == expected