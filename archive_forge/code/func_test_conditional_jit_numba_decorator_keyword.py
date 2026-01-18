import importlib
from unittest.mock import Mock
import numpy as np
import pytest
from ...stats.stats_utils import stats_variance_2d as svar
from ...utils import Numba, _numba_var, numba_check
from ..helpers import running_on_ci
from .test_utils import utils_with_numba_import_fail  # pylint: disable=unused-import
def test_conditional_jit_numba_decorator_keyword(monkeypatch):
    """Checks else statement and JIT keyword argument"""
    from ... import utils
    numba_mock = Mock()
    monkeypatch.setattr(utils.importlib, 'import_module', lambda x: numba_mock)

    def jit(**kwargs):
        """overwrite numba.jit function"""
        return lambda fn: lambda: (fn(), kwargs)
    numba_mock.jit = jit

    @utils.conditional_jit(keyword_argument='A keyword argument')
    def placeholder_func():
        """This function does nothing"""
        return 'output'
    function_results, wrapper_result = placeholder_func()
    assert wrapper_result == {'keyword_argument': 'A keyword argument', 'nopython': True}
    assert function_results == 'output'