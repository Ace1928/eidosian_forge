from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_from_dict(self):
    owner = {'user_id': 'Luke', 'tenant_id': 'Skywalker'}
    ctx = context.Context.from_dict(owner)
    self.assertEqual(owner['user_id'], ctx.user_id)
    self.assertEqual(owner['tenant_id'], ctx.tenant_id)