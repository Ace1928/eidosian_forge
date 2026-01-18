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
def test_decode_no_key_id(token_factory):
    token = token_factory(key_id=False)
    certs = {'2': PUBLIC_CERT_BYTES}
    payload = jwt.decode(token, certs)
    assert payload['user'] == 'billy bob'