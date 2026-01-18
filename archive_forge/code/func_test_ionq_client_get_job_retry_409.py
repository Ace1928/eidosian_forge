import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.get')
def test_ionq_client_get_job_retry_409(mock_get):
    response1 = mock.MagicMock()
    response2 = mock.MagicMock()
    mock_get.side_effect = [response1, response2]
    response1.ok = False
    response1.status_code = requests.codes.conflict
    response1.request.method = 'GET'
    response2.ok = True
    response2.json.return_value = {'foo': 'bar'}
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart')
    response = client.get_job(job_id='job_id')
    assert response == {'foo': 'bar'}
    expected_headers = {'Authorization': 'apiKey to_my_heart', 'Content-Type': 'application/json', 'User-Agent': client._user_agent()}
    mock_get.assert_called_with('http://example.com/v0.3/jobs/job_id', headers=expected_headers)