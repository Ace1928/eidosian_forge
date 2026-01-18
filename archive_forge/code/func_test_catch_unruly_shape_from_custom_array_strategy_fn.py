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
@given(st.data())
def test_catch_unruly_shape_from_custom_array_strategy_fn(self, data):

    def dodgy_array_strategy_fn(*, shape=None, dtype=None):
        """Dodgy function which ignores the shape it was passed"""
        return npst.arrays(shape=(3, 2), dtype=dtype)
    with pytest.raises(ValueError, match='returned an array object with a different shape'):
        data.draw(variables(array_strategy_fn=dodgy_array_strategy_fn, dims=st.just({'a': 2, 'b': 1}), dtype=supported_dtypes()))