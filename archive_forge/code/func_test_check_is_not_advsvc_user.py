from unittest import mock
from neutron_lib import context
from neutron_lib.policy import _engine as policy_engine
from neutron_lib.tests import _base as base
def test_check_is_not_advsvc_user(self):
    ctx = context.Context('me', 'my_project', roles=['user'])
    self.assertFalse(policy_engine.check_is_advsvc(ctx))