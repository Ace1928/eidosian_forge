import functools
import unittest
from pecan import expose
from pecan import util
from pecan.compat import getargspec
def test_class_based_decorator(self):

    class deco(object):

        def __init__(self, arg):
            self.arg = arg

        def __call__(self, f):

            @functools.wraps(f)
            def wrapper(*args, **kw):
                assert self.arg == '12345'
                return f(*args, **kw)
            return wrapper

    class RootController(object):

        @expose()
        @deco('12345')
        def index(self, a, b, c):
            return 'Hello, World!'
    argspec = util._cfg(RootController.index)['argspec']
    assert argspec.args == ['self', 'a', 'b', 'c']