import base64
import ctypes
import os
import mock
import pytest  # type: ignore
from requests.packages.urllib3.util.ssl_ import create_urllib3_context  # type: ignore
import urllib3.contrib.pyopenssl  # type: ignore
from google.auth import exceptions
from google.auth.transport import _custom_tls_signer
def test__compute_sha256_digest():
    to_be_signed = ctypes.create_string_buffer(b'foo')
    sig = _custom_tls_signer._compute_sha256_digest(to_be_signed, 4)
    assert base64.b64encode(sig).decode() == 'RG5gyEH8CAAh3lxgbt2PLPAHPO8p6i9+cn5dqHfUUYM='