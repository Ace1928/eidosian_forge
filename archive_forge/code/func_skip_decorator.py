import sys
import os
import tempfile
import unittest
from ..py3compat import string_types, which
def skip_decorator(f):
    import nose
    if callable(skip_condition):
        skip_val = skip_condition
    else:
        skip_val = lambda: skip_condition

    def get_msg(func, msg=None):
        """Skip message with information about function being skipped."""
        if msg is None:
            out = 'Test skipped due to test condition.'
        else:
            out = msg
        return 'Skipping test: %s. %s' % (func.__name__, out)

    def skipper_func(*args, **kwargs):
        """Skipper for normal test functions."""
        if skip_val():
            raise nose.SkipTest(get_msg(f, msg))
        else:
            return f(*args, **kwargs)

    def skipper_gen(*args, **kwargs):
        """Skipper for test generators."""
        if skip_val():
            raise nose.SkipTest(get_msg(f, msg))
        else:
            for x in f(*args, **kwargs):
                yield x
    if nose.util.isgenerator(f):
        skipper = skipper_gen
    else:
        skipper = skipper_func
    return nose.tools.make_decorator(f)(skipper)