import base64
import ctypes
import os
import mock
import pytest  # type: ignore
from requests.packages.urllib3.util.ssl_ import create_urllib3_context  # type: ignore
import urllib3.contrib.pyopenssl  # type: ignore
from google.auth import exceptions
from google.auth.transport import _custom_tls_signer
def test_load_signer_lib():
    with mock.patch('ctypes.CDLL', return_value=mock.MagicMock()):
        lib = _custom_tls_signer.load_signer_lib('/path/to/signer/lib')
    assert lib.SignForPython.restype == ctypes.c_int
    assert lib.SignForPython.argtypes == [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    assert lib.GetCertPemForPython.restype == ctypes.c_int
    assert lib.GetCertPemForPython.argtypes == [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]