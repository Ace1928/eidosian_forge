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
@given(st.data(), st.integers(0, 3))
def test_given_array_strat_arbitrary_size_and_arbitrary_data(self, data, ndims):
    dim_names = data.draw(dimension_names(min_dims=ndims, max_dims=ndims))

    def array_strategy_fn(*, shape=None, dtype=None):
        return npst.arrays(shape=shape, dtype=dtype)
    var = data.draw(variables(array_strategy_fn=array_strategy_fn, dims=st.just(dim_names), dtype=supported_dtypes()))
    assert var.ndim == ndims