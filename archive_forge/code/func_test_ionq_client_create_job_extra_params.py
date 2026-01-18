import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.post')
def test_ionq_client_create_job_extra_params(mock_post):
    mock_post.return_value.status_code.return_value = requests.codes.ok
    mock_post.return_value.json.return_value = {'foo': 'bar'}
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart')
    program = ionq.SerializedProgram(body={'job': 'mine'}, metadata={'a': '0,1'}, settings={})
    response = client.create_job(serialized_program=program, repetitions=200, target='qpu', name='bacon', extra_query_params={'error_mitigation': {'debias': True}})
    assert response == {'foo': 'bar'}
    expected_json = {'target': 'qpu', 'lang': 'json', 'body': {'job': 'mine'}, 'name': 'bacon', 'shots': '200', 'error_mitigation': {'debias': True}, 'metadata': {'shots': '200', 'a': '0,1'}}
    expected_headers = {'Authorization': 'apiKey to_my_heart', 'Content-Type': 'application/json', 'User-Agent': client._user_agent()}
    mock_post.assert_called_with('http://example.com/v0.3/jobs', json=expected_json, headers=expected_headers)