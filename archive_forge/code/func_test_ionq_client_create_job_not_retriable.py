import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.post')
def test_ionq_client_create_job_not_retriable(mock_post):
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.conflict
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart', default_target='simulator')
    with pytest.raises(ionq.IonQException, match='Status: 409'):
        _ = client.create_job(serialized_program=ionq.SerializedProgram(body={'job': 'mine'}, metadata={}, settings={}))