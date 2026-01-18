from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_guards_binary_operations():

    class GoodOp(int):
        pass

    class BadOp(int):

        def __add__(self, other):
            assert False
    context = limited(good=GoodOp(1), bad=BadOp(1))
    with pytest.raises(GuardRejection):
        guarded_eval('1 + bad', context)
    with pytest.raises(GuardRejection):
        guarded_eval('bad + 1', context)
    assert guarded_eval('good + 1', context) == 2
    assert guarded_eval('1 + good', context) == 2