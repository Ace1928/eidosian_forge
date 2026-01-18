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
def test_decode_bad_token_not_base64():
    with pytest.raises((ValueError, TypeError)) as excinfo:
        jwt.decode('1.2.3', PUBLIC_CERT_BYTES)
    assert excinfo.match('Incorrect padding|more than a multiple of 4')