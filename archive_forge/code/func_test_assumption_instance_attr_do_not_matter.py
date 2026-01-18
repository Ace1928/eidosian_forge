from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_assumption_instance_attr_do_not_matter():
    """This is semi-specified in Python documentation.

    However, since the specification says 'not guaranteed
    to work' rather than 'is forbidden to work', future
    versions could invalidate this assumptions. This test
    is meant to catch such a change if it ever comes true.
    """

    class T:

        def __getitem__(self, k):
            return 'a'

        def __getattr__(self, k):
            return 'a'

    def f(self):
        return 'b'
    t = T()
    t.__getitem__ = f
    t.__getattr__ = f
    assert t[1] == 'a'
    assert t[1] == 'a'