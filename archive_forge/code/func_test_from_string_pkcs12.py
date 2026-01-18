import json
import os
from cryptography.hazmat.primitives.asymmetric import rsa
import pytest  # type: ignore
from google.auth import _helpers
from google.auth.crypt import _cryptography_rsa
from google.auth.crypt import base
def test_from_string_pkcs12(self):
    with pytest.raises(ValueError):
        _cryptography_rsa.RSASigner.from_string(PKCS12_KEY_BYTES)