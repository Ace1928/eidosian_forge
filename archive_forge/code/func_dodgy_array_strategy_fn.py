import numpy as np
import numpy.testing as npt
import pytest
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import given
from hypothesis.extra.array_api import make_strategies_namespace
from xarray.core.variable import Variable
from xarray.testing.strategies import (
from xarray.tests import requires_numpy_array_api
def dodgy_array_strategy_fn(*, shape=None, dtype=None):
    """Dodgy function which ignores the shape it was passed"""
    return npst.arrays(shape=(3, 2), dtype=dtype)