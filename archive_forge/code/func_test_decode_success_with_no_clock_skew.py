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
def test_decode_success_with_no_clock_skew(token_factory):
    token = token_factory(claims={'exp': _helpers.datetime_to_secs(_helpers.utcnow() + datetime.timedelta(seconds=1)), 'iat': _helpers.datetime_to_secs(_helpers.utcnow() - datetime.timedelta(seconds=1))})
    jwt.decode(token, PUBLIC_CERT_BYTES)