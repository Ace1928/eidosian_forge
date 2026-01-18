import threading
import time
import warnings
from traits.api import (
from traits.testing.api import UnittestTools
from traits.testing.unittest_tools import unittest
from traits.util.api import deprecated
def test_assert_eventually_true_passes_when_condition_becomes_true(self):

    class A(HasTraits):
        foo = Bool(False)

    def condition(a_object):
        return a_object.foo
    a = A()

    def thread_target(a):
        time.sleep(1.0)
        a.foo = True
    t = threading.Thread(target=thread_target, args=(a,))
    t.start()
    self.assertEventuallyTrue(condition=condition, obj=a, trait='foo', timeout=10.0)
    t.join()