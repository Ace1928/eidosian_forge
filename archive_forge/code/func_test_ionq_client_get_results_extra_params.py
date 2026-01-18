import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.get')
def test_ionq_client_get_results_extra_params(mock_get):
    mock_get.return_value.ok = True
    mock_get.return_value.json.return_value = {'foo': 'bar'}
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart')
    response = client.get_results(job_id='job_id', extra_query_params={'sharpen': False})
    assert response == {'foo': 'bar'}
    expected_headers = {'Authorization': 'apiKey to_my_heart', 'Content-Type': 'application/json', 'User-Agent': client._user_agent()}
    mock_get.assert_called_with('http://example.com/v0.3/jobs/job_id/results', headers=expected_headers, params={'sharpen': False})