import numpy.testing as npt
from numpy.testing import assert_allclose
import numpy as np
import pytest
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distdiscrete, invdistdiscrete
from scipy.stats._distn_infrastructure import rv_discrete_frozen
@pytest.mark.parametrize('distname, shapename, shapes', cases_test_discrete_integer_shapes())
def test_integer_shapes(distname, shapename, shapes):
    dist = getattr(stats, distname)
    shape_info = dist._shape_info()
    shape_names = [shape.name for shape in shape_info]
    i = shape_names.index(shapename)
    shapes_copy = list(shapes)
    valid_shape = shapes[i]
    invalid_shape = valid_shape - 0.5
    new_valid_shape = valid_shape - 1
    shapes_copy[i] = [[valid_shape], [invalid_shape], [new_valid_shape]]
    a, b = dist.support(*shapes)
    x = np.round(np.linspace(a, b, 5))
    pmf = dist.pmf(x, *shapes_copy)
    assert not np.any(np.isnan(pmf[0, :]))
    assert np.all(np.isnan(pmf[1, :]))
    assert not np.any(np.isnan(pmf[2, :]))