import unittest
from unittest import mock
from traits.api import (
from traits.observation import _has_traits_helpers as helpers
from traits.observation import expression
from traits.observation.observe import observe
def test_dict_equality_prevent_change_event(self):
    instance = ObjectWithEqualityComparisonMode()
    instance.dict_values = {'1': 1}
    handler = mock.Mock()
    observe(object=instance, expression=expression.trait('dict_values').dict_items(), handler=handler)
    instance.dict_values = {'1': 1}
    self.assertEqual(handler.call_count, 0)
    instance.dict_values['2'] = 2
    self.assertEqual(handler.call_count, 1)