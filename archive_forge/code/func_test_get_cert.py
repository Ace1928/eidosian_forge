import base64
import ctypes
import os
import mock
import pytest  # type: ignore
from requests.packages.urllib3.util.ssl_ import create_urllib3_context  # type: ignore
import urllib3.contrib.pyopenssl  # type: ignore
from google.auth import exceptions
from google.auth.transport import _custom_tls_signer
def test_get_cert():
    mock_cert_len = 10
    mock_signer_lib = mock.MagicMock()
    mock_signer_lib.GetCertPemForPython.return_value = mock_cert_len
    mock_cert = _custom_tls_signer.get_cert(mock_signer_lib, FAKE_ENTERPRISE_CERT_FILE_PATH)
    assert mock_signer_lib.GetCertPemForPython.call_count == 2
    assert len(mock_cert) == mock_cert_len