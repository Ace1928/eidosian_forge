import unittest
from unittest import mock
from traits.api import (
from traits.observation import _has_traits_helpers as helpers
from traits.observation import expression
from traits.observation.observe import observe
def test_list_equality_prevent_change_event(self):
    instance = ObjectWithEqualityComparisonMode()
    instance.list_values = [1]
    handler = mock.Mock()
    observe(object=instance, expression=expression.trait('list_values').list_items(), handler=handler)
    instance.list_values = [1]
    self.assertEqual(handler.call_count, 0)
    instance.list_values.append(2)
    self.assertEqual(handler.call_count, 1)