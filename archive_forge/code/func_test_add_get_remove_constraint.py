from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_add_get_remove_constraint(self):
    ctx = context.Context('user_id', 'tenant_id')
    self.assertIsNone(ctx.get_transaction_constraint())
    ctx.set_transaction_constraint('networks', 'net_id', 44)
    constraint = ctx.get_transaction_constraint()
    self.assertEqual(44, constraint.if_revision_match)
    self.assertEqual('networks', constraint.resource)
    self.assertEqual('net_id', constraint.resource_id)
    ctx.clear_transaction_constraint()
    self.assertIsNone(ctx.get_transaction_constraint())