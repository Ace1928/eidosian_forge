import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
def test_ionq_client_invalid_target():
    with pytest.raises(AssertionError, match='the store'):
        _ = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='a', default_target='the store')
    with pytest.raises(AssertionError, match='Target'):
        _ = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='a', default_target='the store')