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
def test_encode_basic(signer):
    test_payload = {'test': 'value'}
    encoded = jwt.encode(signer, test_payload)
    header, payload, _, _ = jwt._unverified_decode(encoded)
    assert payload == test_payload
    assert header == {'typ': 'JWT', 'alg': 'RS256', 'kid': signer.key_id}