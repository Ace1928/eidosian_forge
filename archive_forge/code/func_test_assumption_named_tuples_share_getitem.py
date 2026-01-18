from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_assumption_named_tuples_share_getitem():
    """Check assumption on named tuples sharing __getitem__"""
    from typing import NamedTuple

    class A(NamedTuple):
        pass

    class B(NamedTuple):
        pass
    assert A.__getitem__ == B.__getitem__