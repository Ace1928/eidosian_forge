import unittest
from traits.api import (
from traits.observation.api import (
def test_items_on_a_list_not_observable_by_named_trait(self):
    team = Team()
    team.observe(lambda e: None, trait('member_names').list_items().trait('does_not_exist'))
    with self.assertRaises(ValueError) as exception_cm:
        team.member_names = ['Paul']
    self.assertEqual(str(exception_cm.exception), "Trait named 'does_not_exist' not found on 'Paul'.")