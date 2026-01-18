from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_bind_self(self):

    class F:

        def f(a, self):
            return (a, self)
    an_f = F()
    partial_f = functools.partial(F.f, an_f)
    ba = inspect.signature(partial_f).bind(self=10)
    self.assertEqual((an_f, 10), partial_f(*ba.args, **ba.kwargs))