import functools
import warnings
import threading
import sys
def returns(rtype):
    """Decorator to ensure that the decorated function returns the given
    type as argument.

    Example:
        @accepts(int, (int,float))
        @returns((int,float))
        def func(arg1, arg2):
            return arg1 * arg2
    """

    def check_returns(f):

        def new_f(*args, **kwds):
            result = f(*args, **kwds)
            assert isinstance(result, rtype), 'return value %r does not match %s' % (result, rtype)
            return result
        new_f.__name__ = f.__name__
        return new_f
    return check_returns