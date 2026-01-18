import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.get')
def test_ionq_client_list_calibrations_retry_nonstandard_error(mock_get):
    response1 = mock.MagicMock()
    response2 = mock.MagicMock()
    mock_get.side_effect = [response1, response2]
    response1.ok = False
    response1.status_code = 524
    response2.ok = True
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart', default_target='simulator')
    client.list_calibrations()
    assert mock_get.call_count == 2