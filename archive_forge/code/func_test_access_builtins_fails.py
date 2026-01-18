from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_access_builtins_fails():
    context = limited()
    with pytest.raises(NameError):
        guarded_eval('this_is_not_builtin', context)