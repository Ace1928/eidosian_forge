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
def test_encode_extra_headers(signer):
    encoded = jwt.encode(signer, {}, header={'extra': 'value'})
    header = jwt.decode_header(encoded)
    assert header == {'typ': 'JWT', 'alg': 'RS256', 'kid': signer.key_id, 'extra': 'value'}