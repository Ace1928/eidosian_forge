import json
import os
import mock
from pyasn1_modules import pem  # type: ignore
import pytest  # type: ignore
import rsa  # type: ignore
import six
from google.auth import _helpers
from google.auth.crypt import _python_rsa
from google.auth.crypt import base
def test_from_string_pub_cert_failure(self):
    cert_bytes = PUBLIC_CERT_BYTES
    true_der = rsa.pem.load_pem(cert_bytes, 'CERTIFICATE')
    load_pem_patch = mock.patch('rsa.pem.load_pem', return_value=true_der + b'extra', autospec=True)
    with load_pem_patch as load_pem:
        with pytest.raises(ValueError):
            _python_rsa.RSAVerifier.from_string(cert_bytes)
        load_pem.assert_called_once_with(cert_bytes, 'CERTIFICATE')