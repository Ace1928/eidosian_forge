import threading
import time
import warnings
from traits.api import (
from traits.testing.api import UnittestTools
from traits.testing.unittest_tools import unittest
from traits.util.api import deprecated
def test_assert_eventually_true_fails_on_timeout(self):

    class A(HasTraits):
        foo = Bool(False)
    a = A()

    def condition(a_object):
        return a_object.foo
    with self.assertRaises(self.failureException):
        self.assertEventuallyTrue(condition=condition, obj=a, trait='foo', timeout=1.0)