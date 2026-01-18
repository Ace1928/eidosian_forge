from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_get_admin_context_not_update_local_store(self):
    ctx = context.Context('user_id', 'tenant_id')
    req_id_before = oslo_context.get_current().request_id
    self.assertEqual(ctx.request_id, req_id_before)
    ctx_admin = context.get_admin_context()
    self.assertEqual(req_id_before, oslo_context.get_current().request_id)
    self.assertNotEqual(req_id_before, ctx_admin.request_id)