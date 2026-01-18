from datetime import (
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
@pytest.fixture(params=[pytest.param(('numba', True), marks=[td.skip_if_no('numba'), pytest.mark.single_cpu]), ('cython', True), ('cython', False)])
def engine_and_raw(request):
    """engine and raw keyword arguments for rolling.apply"""
    return request.param