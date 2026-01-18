import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.delete')
def test_ionq_client_delete_job_unauthorized(mock_delete):
    mock_delete.return_value.ok = False
    mock_delete.return_value.status_code = requests.codes.unauthorized
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart', default_target='simulator')
    with pytest.raises(ionq.IonQException, match='Not authorized'):
        client.delete_job('job_id')