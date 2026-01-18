import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.get')
def test_ionq_client_list_jobs_batches(mock_get):
    mock_get.return_value.ok = True
    mock_get.return_value.json.side_effect = [{'jobs': [{'id': '1'}], 'next': 'a'}, {'jobs': [{'id': '2'}], 'next': 'b'}, {'jobs': [{'id': '3'}]}]
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart')
    response = client.list_jobs(batch_size=1)
    assert response == [{'id': '1'}, {'id': '2'}, {'id': '3'}]
    expected_headers = {'Authorization': 'apiKey to_my_heart', 'Content-Type': 'application/json', 'User-Agent': client._user_agent()}
    url = 'http://example.com/v0.3/jobs'
    mock_get.assert_has_calls([mock.call(url, headers=expected_headers, json={'limit': 1}, params={}), mock.call().json(), mock.call(url, headers=expected_headers, json={'limit': 1}, params={'next': 'a'}), mock.call().json(), mock.call(url, headers=expected_headers, json={'limit': 1}, params={'next': 'b'}), mock.call().json()])