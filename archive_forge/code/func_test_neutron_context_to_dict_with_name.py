from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_to_dict_with_name(self):
    ctx = context.Context('user_id', 'tenant_id', tenant_name='tenant_name', user_name='user_name')
    ctx_dict = ctx.to_dict()
    self.assertEqual('user_name', ctx_dict['user_name'])
    self.assertEqual('tenant_name', ctx_dict['tenant_name'])
    self.assertEqual('tenant_name', ctx_dict['project_name'])