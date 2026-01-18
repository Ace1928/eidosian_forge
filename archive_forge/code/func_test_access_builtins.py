from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@pytest.mark.parametrize('context', MINIMAL_OR_HIGHER)
def test_access_builtins(context):
    assert guarded_eval('round', context()) == round