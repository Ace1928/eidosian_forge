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
def fixed_array_strategy_fn(*, shape=None, dtype=None):
    """The fact this ignores shape and dtype is only okay because compatible shape & dtype will be passed separately."""
    return st.just(arr)