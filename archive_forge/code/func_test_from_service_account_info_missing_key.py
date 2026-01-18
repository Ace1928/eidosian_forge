import base64
import json
import os
from cryptography.hazmat.primitives.asymmetric import ec
import pytest  # type: ignore
from google.auth import _helpers
from google.auth.crypt import base
from google.auth.crypt import es256
def test_from_service_account_info_missing_key(self):
    with pytest.raises(ValueError) as excinfo:
        es256.ES256Signer.from_service_account_info({})
    assert excinfo.match(base._JSON_FILE_PRIVATE_KEY)