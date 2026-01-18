from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_create_with_timestamp(self):
    now = 'Right Now!'
    ctx = context.Context('user_id', 'tenant_id', timestamp=now)
    self.assertEqual(now, ctx.timestamp)