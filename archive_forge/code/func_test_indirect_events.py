import threading
import time
import warnings
from traits.api import (
from traits.testing.api import UnittestTools
from traits.testing.unittest_tools import unittest
from traits.util.api import deprecated
def test_indirect_events(self):
    """ Check catching indirect change events.
        """
    test_object = self.test_object
    with self.assertTraitChanges(test_object, 'list_of_numbers[]') as result:
        test_object.flag = True
        test_object.number = -3.0
    expected = (test_object, 'list_of_numbers_items', [], [-3.0])
    self.assertSequenceEqual(expected, result.event)