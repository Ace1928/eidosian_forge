import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.get')
def test_ionq_client_list_jobs(mock_get):
    mock_get.return_value.ok = True
    mock_get.return_value.json.return_value = {'jobs': [{'id': '1'}, {'id': '2'}]}
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart')
    response = client.list_jobs()
    assert response == [{'id': '1'}, {'id': '2'}]
    expected_headers = {'Authorization': 'apiKey to_my_heart', 'Content-Type': 'application/json', 'User-Agent': client._user_agent()}
    mock_get.assert_called_with('http://example.com/v0.3/jobs', headers=expected_headers, json={'limit': 1000}, params={})