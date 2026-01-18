import json
import os
import pytest  # type: ignore
import six
from google.auth import _service_account_info
from google.auth import crypt
def test_from_dict_es256_signer():
    signer = _service_account_info.from_dict(GDCH_SERVICE_ACCOUNT_INFO, use_rsa_signer=False)
    assert isinstance(signer, crypt.ES256Signer)
    assert signer.key_id == GDCH_SERVICE_ACCOUNT_INFO['private_key_id']