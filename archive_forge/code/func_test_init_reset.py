from unittest import mock
from neutron_lib import context
from neutron_lib.policy import _engine as policy_engine
from neutron_lib.tests import _base as base
def test_init_reset(self):
    self.assertIsNone(policy_engine._ROLE_ENFORCER)
    policy_engine.init()
    self.assertIsNotNone(policy_engine._ROLE_ENFORCER)