import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.post')
def test_ionq_client_create_job_default_target(mock_post):
    mock_post.return_value.status_code.return_value = requests.codes.ok
    mock_post.return_value.json.return_value = {'foo'}
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart', default_target='simulator')
    _ = client.create_job(ionq.SerializedProgram(body={'job': 'mine'}, metadata={}, settings={}))
    assert mock_post.call_args[1]['json']['target'] == 'simulator'