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
def test_bad_signing_key_v4(self):
    creds_ref = {'secret': u'e7a7a2240136494986991a6598d9fb9f'}
    credentials = {'token': 'QVdTNC1ITUFDLVNIQTI1NgoyMDE1MDgyNFQxMTIwNDFaCjIwMTUwODI0L1JlZ2lvbk9uZS9zMy9hd3M0X3JlcXVlc3QKZjIyMTU1ODBlZWI5YTE2NzM1MWJkOTNlODZjM2I2ZjA0YTkyOGY1YzU1MjBhMzkzNWE0NTM1NDBhMDk1NjRiNQ==', 'signature': '52d02211a3767d00b2104ab28c9859003b0e9c8735cd10de7975f3b1212cca41'}
    self.assertRaises(exception.Unauthorized, s3tokens.S3Resource._check_signature, creds_ref, credentials)