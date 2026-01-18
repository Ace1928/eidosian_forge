import base64
import hashlib
import hmac
import uuid
import http.client
from keystone.api import s3tokens
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_bad_token_v4(self):
    creds_ref = {'secret': u'e7a7a2240136494986991a6598d9fb9f'}
    credentials = {'token': 'QVdTNC1BQUEKWApYClg=', 'signature': ''}
    self.assertRaises(exception.Unauthorized, s3tokens.S3Resource._check_signature, creds_ref, credentials)
    credentials = {'token': 'QVdTNC1ITUFDLVNIQTI1NgpYCi8vczMvYXdzTl9yZXF1ZXN0Clg=', 'signature': ''}
    self.assertRaises(exception.Unauthorized, s3tokens.S3Resource._check_signature, creds_ref, credentials)