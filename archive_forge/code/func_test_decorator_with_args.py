import functools
import unittest
from pecan import expose
from pecan import util
from pecan.compat import getargspec
def test_decorator_with_args(self):

    def dec(flag):

        def inner(f):

            @functools.wraps(f)
            def wrapped(*a, **kw):
                return f(*a, **kw)
            return wrapped
        return inner
    expected = getargspec(self.controller.index.__func__)
    actual = util.getargspec(dec(True)(self.controller.index.__func__))
    assert expected == actual
    expected = getargspec(self.controller.static_index)
    actual = util.getargspec(dec(True)(self.controller.static_index))
    assert expected == actual