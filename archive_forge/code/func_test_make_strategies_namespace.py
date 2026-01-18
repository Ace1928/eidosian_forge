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
@requires_numpy_array_api
@given(st.data())
def test_make_strategies_namespace(self, data):
    """
        Test not causing a hypothesis.InvalidArgument by generating a dtype that's not in the array API.

        We still want to generate dtypes not in the array API by default, but this checks we don't accidentally override
        the user's choice of dtypes with non-API-compliant ones.
        """
    from numpy import array_api as np_array_api
    np_array_api_st = make_strategies_namespace(np_array_api)
    data.draw(variables(array_strategy_fn=np_array_api_st.arrays, dtype=np_array_api_st.scalar_dtypes()))