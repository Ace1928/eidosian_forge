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
def test_decode_payload_object(signer):
    payload = jwt.encode(signer, 'iatexp')
    with pytest.raises(ValueError) as excinfo:
        jwt.decode(payload, certs=PUBLIC_CERT_BYTES)
    assert excinfo.match('Payload segment should be a JSON object: ' + str(b'ImlhdGV4cCI'))