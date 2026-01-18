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
def test_decode_valid_unverified(token_factory):
    payload = jwt.decode(token_factory(), certs=OTHER_CERT_BYTES, verify=False)
    assert payload['aud'] == 'audience@example.com'
    assert payload['user'] == 'billy bob'
    assert payload['metadata']['meta'] == 'data'