from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_admin_to_dict(self):
    self.db_api_session.return_value = 'fakesession'
    ctx = context.get_admin_context()
    ctx_dict = ctx.to_dict()
    self.assertIsNone(ctx_dict['user_id'])
    self.assertIsNone(ctx_dict['tenant_id'])
    self.assertIsNone(ctx_dict['auth_token'])
    self.assertTrue(ctx_dict['is_admin'])
    self.assertIn('admin', ctx_dict['roles'])
    self.assertIsNotNone(ctx.session)
    self.assertNotIn('session', ctx_dict)