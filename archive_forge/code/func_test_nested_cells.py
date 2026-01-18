import functools
import unittest
from pecan import expose
from pecan import util
from pecan.compat import getargspec
def test_nested_cells(self):

    def before(handler):

        def deco(f):

            def wrapped(*args, **kwargs):
                if callable(handler):
                    handler()
                return f(*args, **kwargs)
            return wrapped
        return deco

    class RootController(object):

        @expose()
        @before(lambda: True)
        def index(self, a, b, c):
            return 'Hello, World!'
    argspec = util._cfg(RootController.index)['argspec']
    assert argspec.args == ['self', 'a', 'b', 'c']