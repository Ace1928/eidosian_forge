import numpy as np
import pytest
from pandas import (
from pandas.core.groupby.base import (
@pytest.fixture(params=[False])
def nogil(request):
    """nogil keyword argument for numba.jit"""
    return request.param