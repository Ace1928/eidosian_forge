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
def test_decode_no_cert(token_factory):
    certs = {'2': PUBLIC_CERT_BYTES}
    with pytest.raises(ValueError) as excinfo:
        jwt.decode(token_factory(), certs)
    assert excinfo.match('Certificate for key id 1 not found')