from glance.api import policy
from glance.common import exception
from glance.common import property_utils
import glance.context
from glance.tests.unit import base
def test_check_property_rules_read_none(self):
    self.rules_checker = property_utils.PropertyRules()
    self.assertTrue(self.rules_checker.check_property_rules('x_none_read', 'create', create_context(self.policy, ['admin', 'member'])))
    self.assertFalse(self.rules_checker.check_property_rules('x_none_read', 'read', create_context(self.policy, [''])))
    self.assertFalse(self.rules_checker.check_property_rules('x_none_read', 'update', create_context(self.policy, [''])))
    self.assertFalse(self.rules_checker.check_property_rules('x_none_read', 'delete', create_context(self.policy, [''])))