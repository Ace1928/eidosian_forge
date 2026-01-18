import base64
import datetime
import json
import os
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import crypt
from google.auth import exceptions
from google.auth import jwt
def test_decode_bad_token_not_json():
    token = b'.'.join([base64.urlsafe_b64encode(b'123!')] * 3)
    with pytest.raises(ValueError) as excinfo:
        jwt.decode(token, PUBLIC_CERT_BYTES)
    assert excinfo.match("Can\\'t parse segment")