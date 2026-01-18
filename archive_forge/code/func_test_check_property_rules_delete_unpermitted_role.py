from glance.api import policy
from glance.common import exception
from glance.common import property_utils
import glance.context
from glance.tests.unit import base
def test_check_property_rules_delete_unpermitted_role(self):
    self.rules_checker = property_utils.PropertyRules(self.policy)
    self.assertFalse(self.rules_checker.check_property_rules('test_prop', 'delete', create_context(self.policy, ['member'])))