import base64
import json
import os
from cryptography.hazmat.primitives.asymmetric import ec
import pytest  # type: ignore
from google.auth import _helpers
from google.auth.crypt import base
from google.auth.crypt import es256
def test_from_string_bogus_key(self):
    key_bytes = 'bogus-key'
    with pytest.raises(ValueError):
        es256.ES256Signer.from_string(key_bytes)