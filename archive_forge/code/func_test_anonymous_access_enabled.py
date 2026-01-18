import webob
from glance.api.middleware import context
import glance.context
from glance.tests.unit import base
def test_anonymous_access_enabled(self):
    req = self._build_request(identity_status='Nope')
    self.config(allow_anonymous_access=True)
    middleware = self._build_middleware()
    middleware.process_request(req)
    self.assertIsNone(req.context.auth_token)
    self.assertIsNone(req.context.user_id)
    self.assertIsNone(req.context.project_id)
    self.assertEqual([], req.context.roles)
    self.assertFalse(req.context.is_admin)
    self.assertTrue(req.context.read_only)