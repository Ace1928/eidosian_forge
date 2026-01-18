from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_from_dict_validate(self):
    ctx = context.Context(user_id='user_id1', user_name='user', tenant_id='tenant_id1', tenant_name='project1', request_id='request_id1', auth_token='auth_token1')
    values = {'user_id': 'user_id1', 'user_name': 'user', 'tenant_id': 'tenant_id1', 'tenant_name': 'project1', 'request_id': 'request_id1', 'auth_token': 'auth_token1'}
    ctx2 = context.Context.from_dict(values)
    ctx_dict = ctx.to_dict()
    ctx_dict.pop('timestamp')
    ctx2_dict = ctx2.to_dict()
    ctx2_dict.pop('timestamp')
    self.assertEqual(ctx_dict, ctx2_dict)