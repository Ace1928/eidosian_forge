import threading
import time
import warnings
from traits.api import (
from traits.testing.api import UnittestTools
from traits.testing.unittest_tools import unittest
from traits.util.api import deprecated
def test_exception_inside_context(self):
    """ Check that exception inside the context statement block are
        propagated.

        """
    test_object = self.test_object
    with self.assertRaises(AttributeError):
        with self.assertTraitChanges(test_object, 'number'):
            test_object.i_do_exist
    with self.assertRaises(AttributeError):
        with self.assertTraitDoesNotChange(test_object, 'number'):
            test_object.i_do_exist