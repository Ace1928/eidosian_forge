import uuid
from oauthlib import oauth1
from testtools import matchers
from keystoneauth1.extras import oauth1 as ksa_oauth1
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils as test_utils
def stub_auth(self, subject_token=None, **kwargs):
    if not subject_token:
        subject_token = self.TEST_TOKEN
    self.stub_url('POST', ['auth', 'tokens'], headers={'X-Subject-Token': subject_token}, **kwargs)