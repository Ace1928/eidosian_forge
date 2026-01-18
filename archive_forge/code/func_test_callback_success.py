import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
def test_callback_success(self):
    callback = mock.Mock()
    callback.return_value = (pytest.public_cert_bytes, pytest.private_key_bytes)
    found_cert_key, cert, key = _mtls_helper.get_client_cert_and_key(callback)
    assert found_cert_key
    assert cert == pytest.public_cert_bytes
    assert key == pytest.private_key_bytes