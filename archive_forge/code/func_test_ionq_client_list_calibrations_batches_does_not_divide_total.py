import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.get')
def test_ionq_client_list_calibrations_batches_does_not_divide_total(mock_get):
    mock_get.return_value.ok = True
    mock_get.return_value.json.side_effect = [{'calibrations': [{'id': '1'}, {'id': '2'}], 'next': 'a'}, {'calibrations': [{'id': '3'}]}]
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart')
    response = client.list_calibrations(batch_size=2)
    assert response == [{'id': '1'}, {'id': '2'}, {'id': '3'}]
    expected_headers = {'Authorization': 'apiKey to_my_heart', 'Content-Type': 'application/json', 'User-Agent': client._user_agent()}
    url = 'http://example.com/v0.3/calibrations'
    mock_get.assert_has_calls([mock.call(url, headers=expected_headers, json={'limit': 2}, params={}), mock.call().json(), mock.call(url, headers=expected_headers, json={'limit': 2}, params={'next': 'a'}), mock.call().json()])