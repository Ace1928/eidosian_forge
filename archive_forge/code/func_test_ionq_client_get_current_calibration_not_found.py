import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.get')
def test_ionq_client_get_current_calibration_not_found(mock_get):
    mock_get.return_value.ok = False
    mock_get.return_value.status_code = requests.codes.not_found
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart')
    with pytest.raises(ionq.IonQNotFoundException, match='not find'):
        _ = client.get_current_calibration()