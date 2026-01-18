import threading
import time
import warnings
from traits.api import (
from traits.testing.api import UnittestTools
from traits.testing.unittest_tools import unittest
from traits.util.api import deprecated
def test_change_on_failure(self):
    """ Check behaviour when assertion should be raised for trait change.
        """
    test_object = self.test_object
    with self.assertRaises(AssertionError):
        with self.assertTraitChanges(test_object, 'number') as result:
            test_object.flag = True
    self.assertEqual(result.events, [])
    with self.assertRaises(AssertionError):
        with self.assertTraitChanges(test_object, 'number', count=3) as result:
            test_object.flag = True
            test_object.add_to_number(10.0)
            test_object.add_to_number(10.0)
    expected = [(test_object, 'number', 2.0, 12.0), (test_object, 'number', 12.0, 22.0)]
    self.assertSequenceEqual(expected, result.events)