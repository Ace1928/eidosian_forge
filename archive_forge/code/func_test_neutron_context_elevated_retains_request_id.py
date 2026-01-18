from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_elevated_retains_request_id(self):
    expected_roles = ['admin', 'member', 'reader']
    ctx = context.Context('user_id', 'tenant_id')
    self.assertFalse(ctx.is_admin)
    req_id_before = ctx.request_id
    elevated_ctx = ctx.elevated()
    self.assertTrue(elevated_ctx.is_admin)
    self.assertEqual(req_id_before, elevated_ctx.request_id)
    for expected_role in expected_roles:
        self.assertIn(expected_role, elevated_ctx.roles)