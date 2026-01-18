import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_everything(self):
    """
        Test references search using complex set of objects.
        """

    class Dummy:

        def method(self):
            pass
    o = Dummy()
    D1 = {(): 'baz', None: 'Quux', o: 'Foosh'}
    L = [None, (), D1, 3]
    T = (L, {}, Dummy())
    D2 = {0: 'foo', 1: 'bar', 2: T}
    i = Dummy()
    i.attr = D2
    m = i.method
    w = weakref.ref(m)
    self.assertIn("().__self__.attr[2][0][2]{'Foosh'}", reflect.objgrep(w, o, reflect.isSame))