import importlib
from unittest.mock import Mock
import numpy as np
import pytest
from ...stats.stats_utils import stats_variance_2d as svar
from ...utils import Numba, _numba_var, numba_check
from ..helpers import running_on_ci
from .test_utils import utils_with_numba_import_fail  # pylint: disable=unused-import
@utils.conditional_jit(keyword_argument='A keyword argument')
def placeholder_func():
    """This function does nothing"""
    return 'output'