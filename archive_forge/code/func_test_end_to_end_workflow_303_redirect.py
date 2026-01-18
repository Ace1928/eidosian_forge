import copy
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneauth1.tests.unit import k2k_fixtures
from keystoneauth1.tests.unit import utils
def test_end_to_end_workflow_303_redirect(self):
    self._mock_k2k_flow_urls(redirect_code=303)
    auth_ref = self.k2kplugin.get_auth_ref(self.session)
    self.assertEqual(k2k_fixtures.UNSCOPED_TOKEN_HEADER, auth_ref.auth_token)