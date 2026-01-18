from glance.api import policy
from glance.common import exception
from glance.common import property_utils
import glance.context
from glance.tests.unit import base
def test_check_property_rules_read_unpermitted_policy(self):
    self.assertFalse(self.rules_checker.check_property_rules('spl_creator_policy', 'read', create_context(self.policy, ['fake-role'])))