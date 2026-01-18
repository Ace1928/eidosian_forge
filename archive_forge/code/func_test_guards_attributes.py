from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_guards_attributes():

    class GoodAttr(float):
        pass

    class BadAttr1(float):

        def __getattr__(self, key):
            assert False

    class BadAttr2(float):

        def __getattribute__(self, key):
            assert False
    context = limited(good=GoodAttr(0.5), bad1=BadAttr1(0.5), bad2=BadAttr2(0.5))
    with pytest.raises(GuardRejection):
        guarded_eval('bad1.as_integer_ratio', context)
    with pytest.raises(GuardRejection):
        guarded_eval('bad2.as_integer_ratio', context)
    assert guarded_eval('good.as_integer_ratio()', context) == (1, 2)