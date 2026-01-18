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
def test_given_incorrect_types(self, data):
    with pytest.raises(TypeError, match='dims must be provided as a'):
        data.draw(variables(dims=['x', 'y']))
    with pytest.raises(TypeError, match='dtype must be provided as a'):
        data.draw(variables(dtype=np.dtype('int32')))
    with pytest.raises(TypeError, match='attrs must be provided as a'):
        data.draw(variables(attrs=dict()))
    with pytest.raises(TypeError, match='Callable'):
        data.draw(variables(array_strategy_fn=np.array([0])))