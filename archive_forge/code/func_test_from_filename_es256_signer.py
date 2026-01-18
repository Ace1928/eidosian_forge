import json
import os
import pytest  # type: ignore
import six
from google.auth import _service_account_info
from google.auth import crypt
def test_from_filename_es256_signer():
    _, signer = _service_account_info.from_filename(GDCH_SERVICE_ACCOUNT_JSON_FILE, use_rsa_signer=False)
    assert isinstance(signer, crypt.ES256Signer)
    assert signer.key_id == GDCH_SERVICE_ACCOUNT_INFO['private_key_id']