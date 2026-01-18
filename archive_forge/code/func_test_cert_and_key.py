import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
def test_cert_and_key(self):
    check_cert_and_key(pytest.public_cert_bytes + pytest.private_key_bytes, pytest.public_cert_bytes, pytest.private_key_bytes)
    check_cert_and_key(pytest.private_key_bytes + pytest.public_cert_bytes, pytest.public_cert_bytes, pytest.private_key_bytes)
    check_cert_and_key(pytest.public_cert_bytes + pytest.public_cert_bytes + pytest.private_key_bytes, pytest.public_cert_bytes + pytest.public_cert_bytes, pytest.private_key_bytes)
    check_cert_and_key(pytest.private_key_bytes + pytest.public_cert_bytes + pytest.public_cert_bytes, pytest.public_cert_bytes + pytest.public_cert_bytes, pytest.private_key_bytes)