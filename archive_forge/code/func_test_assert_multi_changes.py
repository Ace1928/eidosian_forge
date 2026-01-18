import threading
import time
import warnings
from traits.api import (
from traits.testing.api import UnittestTools
from traits.testing.unittest_tools import unittest
from traits.util.api import deprecated
def test_assert_multi_changes(self):
    test_object = self.test_object
    with self.assertMultiTraitChanges([test_object], [], ['flag', 'number', 'list_of_numbers[]']) as results:
        test_object.number = 2.0
    events = list(filter(bool, (result.event for result in results)))
    msg = 'The assertion result is not None: {0}'.format(', '.join(events))
    self.assertFalse(events, msg=msg)
    with self.assertMultiTraitChanges([test_object], ['number', 'list_of_numbers[]'], ['flag']) as results:
        test_object.number = 5.0
    events = list(filter(bool, (result.event for result in results)))
    msg = 'The assertion result is None'
    self.assertTrue(events, msg=msg)