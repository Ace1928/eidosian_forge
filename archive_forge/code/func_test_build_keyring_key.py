import datetime
from unittest import mock
from oslo_utils import timeutils
from keystoneclient import access
from keystoneclient import httpclient
from keystoneclient.tests.unit import utils
from keystoneclient.tests.unit.v2_0 import client_fixtures
from keystoneclient import utils as client_utils
def test_build_keyring_key(self):
    with self.deprecations.expect_deprecations_here():
        cl = httpclient.HTTPClient(username=USERNAME, password=PASSWORD, project_id=TENANT_ID, auth_url=AUTH_URL)
    keyring_key = cl._build_keyring_key(auth_url=AUTH_URL, username=USERNAME, tenant_name=TENANT, tenant_id=TENANT_ID, token=TOKEN)
    self.assertEqual(keyring_key, '%s/%s/%s/%s/%s' % (AUTH_URL, TENANT_ID, TENANT, TOKEN, USERNAME))