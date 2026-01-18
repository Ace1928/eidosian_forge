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
def test_decode_header_object(token_factory):
    payload = token_factory()
    payload = b'M7.' + b'.'.join(payload.split(b'.')[1:])
    with pytest.raises(ValueError) as excinfo:
        jwt.decode(payload, certs=PUBLIC_CERT_BYTES)
    assert excinfo.match('Header segment should be a JSON object: ' + str(b'M7'))