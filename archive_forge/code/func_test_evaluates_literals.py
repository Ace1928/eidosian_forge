from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@pytest.mark.parametrize('code,expected', [['1', 1], ['1.0', 1.0], ['0xdeedbeef', 3740122863], ['True', True], ['None', None], ['{}', {}], ['[]', []]])
@pytest.mark.parametrize('context', MINIMAL_OR_HIGHER)
def test_evaluates_literals(code, expected, context):
    assert guarded_eval(code, context()) == expected