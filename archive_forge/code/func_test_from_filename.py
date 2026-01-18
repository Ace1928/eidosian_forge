import json
import os
import pytest  # type: ignore
import six
from google.auth import _service_account_info
from google.auth import crypt
def test_from_filename():
    info, signer = _service_account_info.from_filename(SERVICE_ACCOUNT_JSON_FILE)
    for key, value in six.iteritems(SERVICE_ACCOUNT_INFO):
        assert info[key] == value
    assert isinstance(signer, crypt.RSASigner)
    assert signer.key_id == SERVICE_ACCOUNT_INFO['private_key_id']