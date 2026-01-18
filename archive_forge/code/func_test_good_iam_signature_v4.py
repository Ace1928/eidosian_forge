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
def test_good_iam_signature_v4(self):
    creds_ref = {'secret': u'e7a7a2240136494986991a6598d9fb9f'}
    credentials = {'token': 'QVdTNC1ITUFDLVNIQTI1NgoyMDE1MDgyNFQxMTIwNDFaCjIwMTUwODI0L1JlZ2lvbk9uZS9pYW0vYXdzNF9yZXF1ZXN0CmYyMjE1NTgwZWViOWExNjczNTFiZDkzZTg2YzNiNmYwNGE5MjhmNWM1NTIwYTM5MzVhNDUzNTQwYTA5NTY0YjU=', 'signature': 'db4e15b3040f6afaa9d9d16002de2fc3425beea0c6ea8c1b2bb674f052030b7d'}
    self.assertIsNone(s3tokens.S3Resource._check_signature(creds_ref, credentials))