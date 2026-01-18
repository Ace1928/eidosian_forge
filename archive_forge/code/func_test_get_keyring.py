import datetime
from unittest import mock
from oslo_utils import timeutils
from keystoneclient import access
from keystoneclient import httpclient
from keystoneclient.tests.unit import utils
from keystoneclient.tests.unit.v2_0 import client_fixtures
from keystoneclient import utils as client_utils
def test_get_keyring(self):
    with self.deprecations.expect_deprecations_here():
        cl = httpclient.HTTPClient(username=USERNAME, password=PASSWORD, project_id=TENANT_ID, auth_url=AUTH_URL, use_keyring=True)
    auth_ref = access.AccessInfo.factory(body=PROJECT_SCOPED_TOKEN)
    future = timeutils.utcnow() + datetime.timedelta(minutes=30)
    auth_ref['token']['expires'] = client_utils.isotime(future)
    self.memory_keyring.password = pickle.dumps(auth_ref)
    self.assertTrue(cl.authenticate())
    self.assertTrue(self.memory_keyring.fetched)