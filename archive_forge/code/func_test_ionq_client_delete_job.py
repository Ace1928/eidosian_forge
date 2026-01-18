import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.delete')
def test_ionq_client_delete_job(mock_delete):
    mock_delete.return_value.ok = True
    mock_delete.return_value.json.return_value = {'foo': 'bar'}
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart')
    response = client.delete_job(job_id='job_id')
    assert response == {'foo': 'bar'}
    expected_headers = {'Authorization': 'apiKey to_my_heart', 'Content-Type': 'application/json', 'User-Agent': client._user_agent()}
    mock_delete.assert_called_with('http://example.com/v0.3/jobs/job_id', headers=expected_headers)