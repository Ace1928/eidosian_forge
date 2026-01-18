import os.path
import ddt
from oslo_config import fixture as config_fixture
from oslo_policy import policy as base_policy
from heat.common import exception
from heat.common import policy
from heat.tests import common
from heat.tests import utils
def test_enforce_creds(self):
    enforcer = policy.Enforcer()
    ctx = utils.dummy_context(roles=['admin'])
    self.assertTrue(enforcer.check_is_admin(ctx))