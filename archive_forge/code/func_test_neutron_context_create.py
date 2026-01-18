from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_create(self):
    ctx = context.Context('user_id', 'tenant_id')
    self.assertEqual('user_id', ctx.user_id)
    self.assertEqual('tenant_id', ctx.project_id)
    self.assertEqual('tenant_id', ctx.tenant_id)
    request_id = ctx.request_id
    if isinstance(request_id, bytes):
        request_id = request_id.decode('utf-8')
    self.assertThat(request_id, matchers.StartsWith('req-'))
    self.assertIsNone(ctx.user_name)
    self.assertIsNone(ctx.tenant_name)
    self.assertIsNone(ctx.project_name)
    self.assertIsNone(ctx.auth_token)