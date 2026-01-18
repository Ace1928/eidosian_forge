import datetime
import hashlib
import http.client
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_utils import timeutils
from keystone.common import provider_api
from keystone.common import utils
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_authenticate_expired_request(self):
    self.config_fixture.config(group='credential', auth_ttl=5)
    signer = ec2_utils.Ec2Signer(self.cred_blob['secret'])
    past = timeutils.utcnow() - datetime.timedelta(minutes=10)
    timestamp = utils.isotime(past)
    credentials = {'access': self.cred_blob['access'], 'secret': self.cred_blob['secret'], 'host': 'localhost', 'verb': 'GET', 'path': '/', 'params': {'SignatureVersion': '2', 'Action': 'Test', 'Timestamp': timestamp}}
    credentials['signature'] = signer.generate(credentials)
    self.post('/ec2tokens', body={'credentials': credentials}, expected_status=http.client.UNAUTHORIZED)