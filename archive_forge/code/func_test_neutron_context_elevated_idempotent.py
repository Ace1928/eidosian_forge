from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_elevated_idempotent(self):
    ctx = context.Context('user_id', 'tenant_id')
    expected_roles = ['admin', 'member', 'reader']
    self.assertFalse(ctx.is_admin)
    elevated_ctx = ctx.elevated()
    self.assertTrue(elevated_ctx.is_admin)
    for expected_role in expected_roles:
        self.assertIn(expected_role, elevated_ctx.roles)
    elevated2_ctx = elevated_ctx.elevated()
    self.assertTrue(elevated2_ctx.is_admin)
    for expected_role in expected_roles:
        self.assertIn(expected_role, elevated_ctx.roles)