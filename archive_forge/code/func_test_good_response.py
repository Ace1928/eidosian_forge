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
def test_good_response(self):
    sts = 'string to sign'
    sig = hmac.new(self.cred_blob['secret'].encode('ascii'), sts.encode('ascii'), hashlib.sha1).digest()
    resp = self.post('/s3tokens', body={'credentials': {'access': self.cred_blob['access'], 'signature': base64.b64encode(sig).strip(), 'token': base64.b64encode(sts.encode('ascii')).strip()}}, expected_status=http.client.OK)
    self.assertValidProjectScopedTokenResponse(resp, self.user, forbid_token_id=True)