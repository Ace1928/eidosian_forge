from __future__ import annotations
import pytest
import dask
from dask.local import finish_task, get_sync, sortkey, start_state_from_dask
from dask.order import order
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_exceptions_propagate():

    class MyException(Exception):

        def __init__(self, a, b):
            self.a = a
            self.b = b

        def __str__(self):
            return 'My Exception!'

    def f():
        raise MyException(1, 2)
    from dask.threaded import get
    try:
        get({'x': (f,)}, 'x')
        assert False
    except MyException as e:
        assert 'My Exception!' in str(e)
        assert 'a' in dir(e)
        assert e.a == 1
        assert e.b == 2