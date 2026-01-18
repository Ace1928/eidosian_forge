from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_to_dict_with_auth_token(self):
    ctx = context.Context('user_id', 'tenant_id', auth_token='auth_token_xxx')
    ctx_dict = ctx.to_dict()
    self.assertEqual('auth_token_xxx', ctx_dict['auth_token'])