import threading
import time
import warnings
from traits.api import (
from traits.testing.api import UnittestTools
from traits.testing.unittest_tools import unittest
from traits.util.api import deprecated
def test_assert_not_deprecated_failures(self):
    with self.assertRaises(self.failureException):
        with self.assertNotDeprecated():
            old_and_dull()