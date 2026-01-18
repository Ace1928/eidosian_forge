import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
def test_instance_adapt_default(self):
    ta = TraitsHolder()
    ta.a_default = SampleAverage()
    self.assertEqual(ta.a_default.get_average(), 200.0)
    self.assertIsInstance(ta.a_default, SampleAverage)
    self.assertFalse(hasattr(ta, 'a_default_'))
    ta.a_default = SampleList()
    self.assertEqual(ta.a_default.get_average(), 20.0)
    self.assertIsInstance(ta.a_default, ListAverageAdapter)
    self.assertFalse(hasattr(ta, 'a_default_'))
    ta.a_default = Sample()
    self.assertEqual(ta.a_default.get_average(), 2.0)
    self.assertIsInstance(ta.a_default, ListAverageAdapter)
    self.assertFalse(hasattr(ta, 'a_default_'))
    ta.a_default = SampleBad()
    self.assertEqual(ta.a_default, None)
    self.assertFalse(hasattr(ta, 'a_default_'))