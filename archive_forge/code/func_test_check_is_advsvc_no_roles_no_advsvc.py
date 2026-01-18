from unittest import mock
from neutron_lib import context
from neutron_lib.policy import _engine as policy_engine
from neutron_lib.tests import _base as base
def test_check_is_advsvc_no_roles_no_advsvc(self):
    policy_engine.init(policy_file='dummy_policy.yaml')
    ctx = context.Context('me', 'my_project', roles=['advsvc'])
    self.assertFalse(policy_engine.check_is_advsvc(ctx))