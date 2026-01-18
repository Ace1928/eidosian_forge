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
def test_encode_custom_alg_in_headers(signer):
    encoded = jwt.encode(signer, {}, header={'alg': 'foo'})
    header = jwt.decode_header(encoded)
    assert header == {'typ': 'JWT', 'alg': 'foo', 'kid': signer.key_id}