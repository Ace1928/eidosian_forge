import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
def test_dynamic_notifiers_methods_failing(self):
    obj = DynamicNotifiers()
    obj.fail = 1
    self.assertCountEqual([0, 1, 2, 3, 4], obj.exceptions_from)
    self.assertEqual([(obj, 'fail', 0, 1)] * 5, self.exceptions)