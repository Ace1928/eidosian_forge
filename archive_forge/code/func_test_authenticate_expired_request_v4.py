import datetime
import hashlib
import http.client
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_utils import timeutils
from keystone.common import provider_api
from keystone.common import utils
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_authenticate_expired_request_v4(self):
    self.config_fixture.config(group='credential', auth_ttl=5)
    signer = ec2_utils.Ec2Signer(self.cred_blob['secret'])
    past = timeutils.utcnow() - datetime.timedelta(minutes=10)
    timestamp = utils.isotime(past)
    hashed_payload = 'GET\n/\nAction=Test\nhost:localhost\nx-amz-date:' + timestamp + '\n\nhost;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    body_hash = hashlib.sha256(hashed_payload.encode()).hexdigest()
    amz_credential = 'AKIAIOSFODNN7EXAMPLE/%s/us-east-1/iam/aws4_request,' % timestamp[:8]
    credentials = {'access': self.cred_blob['access'], 'secret': self.cred_blob['secret'], 'host': 'localhost', 'verb': 'GET', 'path': '/', 'params': {'Action': 'Test', 'X-Amz-Algorithm': 'AWS4-HMAC-SHA256', 'X-Amz-SignedHeaders': 'host,x-amz-date,', 'X-Amz-Credential': amz_credential}, 'headers': {'X-Amz-Date': timestamp}, 'body_hash': body_hash}
    credentials['signature'] = signer.generate(credentials)
    self.post('/ec2tokens', body={'credentials': credentials}, expected_status=http.client.UNAUTHORIZED)