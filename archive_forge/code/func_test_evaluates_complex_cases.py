from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@pytest.mark.parametrize('code,expected', [['(1\n+\n1)', 2], ['list(range(10))[-1:]', [9]], ['list(range(20))[3:-2:3]', [3, 6, 9, 12, 15]]])
@pytest.mark.parametrize('context', LIMITED_OR_HIGHER)
def test_evaluates_complex_cases(code, expected, context):
    assert guarded_eval(code, context()) == expected