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
def test_good_signature_v4(self):
    creds_ref = {'secret': u'e7a7a2240136494986991a6598d9fb9f'}
    credentials = {'token': 'QVdTNC1ITUFDLVNIQTI1NgoyMDE1MDgyNFQxMTIwNDFaCjIwMTUwODI0L1JlZ2lvbk9uZS9zMy9hd3M0X3JlcXVlc3QKZjIyMTU1ODBlZWI5YTE2NzM1MWJkOTNlODZjM2I2ZjA0YTkyOGY1YzU1MjBhMzkzNWE0NTM1NDBhMDk1NjRiNQ==', 'signature': '730ba8f58df6ffeadd78f402e990b2910d60bc5c2aec63619734f096a4dd77be'}
    self.assertIsNone(s3tokens.S3Resource._check_signature(creds_ref, credentials))