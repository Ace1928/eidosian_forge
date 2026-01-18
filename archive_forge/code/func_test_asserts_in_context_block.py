import threading
import time
import warnings
from traits.api import (
from traits.testing.api import UnittestTools
from traits.testing.unittest_tools import unittest
from traits.util.api import deprecated
def test_asserts_in_context_block(self):
    """ Make sure that the traits context manager does not stop
        regular assertions inside the managed code block from happening.
        """
    test_object = TestObject(number=16.0)
    with self.assertTraitDoesNotChange(test_object, 'number'):
        self.assertEqual(test_object.number, 16.0)
    with self.assertRaisesRegex(AssertionError, '16\\.0 != 12\\.0'):
        with self.assertTraitDoesNotChange(test_object, 'number'):
            self.assertEqual(test_object.number, 12.0)