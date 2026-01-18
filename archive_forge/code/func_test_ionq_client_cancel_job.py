import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.put')
def test_ionq_client_cancel_job(mock_put):
    mock_put.return_value.ok = True
    mock_put.return_value.json.return_value = {'foo': 'bar'}
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart')
    response = client.cancel_job(job_id='job_id')
    assert response == {'foo': 'bar'}
    expected_headers = {'Authorization': 'apiKey to_my_heart', 'Content-Type': 'application/json', 'User-Agent': client._user_agent()}
    mock_put.assert_called_with('http://example.com/v0.3/jobs/job_id/status/cancel', headers=expected_headers)