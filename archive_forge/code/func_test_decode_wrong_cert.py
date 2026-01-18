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
def test_decode_wrong_cert(token_factory):
    with pytest.raises(ValueError) as excinfo:
        jwt.decode(token_factory(), OTHER_CERT_BYTES)
    assert excinfo.match('Could not verify token signature')