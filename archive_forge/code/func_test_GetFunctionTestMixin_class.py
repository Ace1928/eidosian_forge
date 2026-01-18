from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_GetFunctionTestMixin_class():

    class TestCustomGetFail(GetFunctionTestMixin):
        get = staticmethod(lambda x, y: 1)
    custom_testget = TestCustomGetFail()
    pytest.raises(AssertionError, custom_testget.test_get)

    class TestCustomGetPass(GetFunctionTestMixin):
        get = staticmethod(get)
    custom_testget = TestCustomGetPass()
    custom_testget.test_get()