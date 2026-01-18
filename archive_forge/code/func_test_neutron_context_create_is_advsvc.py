from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_create_is_advsvc(self):
    ctx = context.Context('user_id', 'tenant_id', is_advsvc=True)
    self.assertFalse(ctx.is_admin)
    self.assertTrue(ctx.is_advsvc)