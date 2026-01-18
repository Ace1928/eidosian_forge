import unittest
from unittest import mock
from traits.api import (
from traits.observation import _has_traits_helpers as helpers
from traits.observation import expression
from traits.observation.observe import observe
def test_basic_trait_equality_prevent_change_event(self):
    instance = ObjectWithEqualityComparisonMode()
    instance.number = 1
    handler = mock.Mock()
    observe(object=instance, expression=expression.trait('number'), handler=handler)
    instance.number = 1.0
    self.assertEqual(handler.call_count, 0)
    instance.number = True
    self.assertEqual(handler.call_count, 0)
    instance.number = 2.0
    self.assertEqual(handler.call_count, 1)
    handler.reset_mock()
    instance.number = CannotCompare()
    self.assertEqual(handler.call_count, 1)