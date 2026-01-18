from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@pytest.mark.parametrize('code,expected', [['-5', -5], ['+5', +5], ['~5', -6]])
@pytest.mark.parametrize('context', LIMITED_OR_HIGHER)
def test_evaluates_unary_operations(code, expected, context):
    assert guarded_eval(code, context()) == expected