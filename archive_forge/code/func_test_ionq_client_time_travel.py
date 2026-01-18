import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
def test_ionq_client_time_travel():
    with pytest.raises(AssertionError, match='time machine'):
        _ = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='a', max_retry_seconds=-1)