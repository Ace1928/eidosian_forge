import itertools
import contextlib
import operator
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_raises, assert_equal
@contextlib.contextmanager
def exc_iter(*args):
    """
    Iterate over Cartesian product of *args, and if an exception is raised,
    add information of the current iterate.
    """
    value = [None]

    def iterate():
        for v in itertools.product(*args):
            value[0] = v
            yield v
    try:
        yield iterate()
    except Exception:
        import traceback
        msg = 'At: %r\n%s' % (repr(value[0]), traceback.format_exc())
        raise AssertionError(msg)