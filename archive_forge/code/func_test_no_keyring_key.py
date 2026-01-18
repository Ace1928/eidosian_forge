import datetime
from unittest import mock
from oslo_utils import timeutils
from keystoneclient import access
from keystoneclient import httpclient
from keystoneclient.tests.unit import utils
from keystoneclient.tests.unit.v2_0 import client_fixtures
from keystoneclient import utils as client_utils
def test_no_keyring_key(self):
    """Test case when no keyring set.

        Ensure that if we don't have use_keyring set in the client that
        the keyring is never accessed.
        """
    with self.deprecations.expect_deprecations_here():
        cl = httpclient.HTTPClient(username=USERNAME, password=PASSWORD, project_id=TENANT_ID, auth_url=AUTH_URL)
    method = 'get_raw_token_from_identity_service'
    with mock.patch.object(cl, method) as meth:
        meth.return_value = (True, PROJECT_SCOPED_TOKEN)
        self.assertTrue(cl.authenticate())
        self.assertEqual(1, meth.call_count)
    self.assertFalse(self.memory_keyring.get_password_called)
    self.assertFalse(self.memory_keyring.set_password_called)