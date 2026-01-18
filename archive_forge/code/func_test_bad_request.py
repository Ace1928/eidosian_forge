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
def test_bad_request(self):
    self.post('/s3tokens', body={}, expected_status=http.client.BAD_REQUEST)
    self.post('/s3tokens', body='not json', expected_status=http.client.BAD_REQUEST)
    self.post('/s3tokens', expected_status=http.client.BAD_REQUEST)