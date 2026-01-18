from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_create_is_service_role(self):
    ctx = context.Context('user_id', 'tenant_id', roles=['service'])
    self.assertFalse(ctx.is_admin)
    self.assertTrue(ctx.is_service_role)