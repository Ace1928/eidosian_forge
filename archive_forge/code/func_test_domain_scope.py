import uuid
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystonemiddleware.auth_token import _base
from keystonemiddleware.tests.unit.auth_token import base
def test_domain_scope(self):
    token_id, token = self.get_token(project=False)
    token.set_domain_scope()
    plugin = self.get_plugin(token_id)
    self.assertEqual(token.domain_id, plugin.user.domain_id)
    self.assertIsNone(plugin.user.project_id)