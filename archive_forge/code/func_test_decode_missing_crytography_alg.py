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
def test_decode_missing_crytography_alg(monkeypatch):
    monkeypatch.delitem(jwt._ALGORITHM_TO_VERIFIER_CLASS, 'ES256')
    headers = json.dumps({u'kid': u'1', u'alg': u'ES256'})
    token = b'.'.join(map(lambda seg: base64.b64encode(seg.encode('utf-8')), [headers, u'{}', u'sig']))
    with pytest.raises(ValueError) as excinfo:
        jwt.decode(token)
    assert excinfo.match('cryptography')