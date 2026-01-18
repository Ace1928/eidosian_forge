import functools
import itertools
import math
import numpy
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from scipy.ndimage._filters import _gaussian_kernel1d
from . import types, float_types, complex_types
@pytest.mark.parametrize('filter_func, size0, size, kwargs', [(ndimage.gaussian_filter, 0, 1.0, kwargs_gauss), (ndimage.uniform_filter, 1, 3, kwargs_other), (ndimage.maximum_filter, 1, 3, kwargs_other), (ndimage.minimum_filter, 1, 3, kwargs_other), (ndimage.median_filter, 1, 3, kwargs_rank), (ndimage.rank_filter, 1, 3, kwargs_rank), (ndimage.percentile_filter, 1, 3, kwargs_rank)])
@pytest.mark.parametrize('axes', itertools.combinations(range(-3, 3), 2))
def test_filter_axes_kwargs(self, filter_func, size0, size, kwargs, axes):
    array = numpy.arange(6 * 8 * 12, dtype=numpy.float64).reshape(6, 8, 12)
    kwargs = {key: numpy.array(val) for key, val in kwargs.items()}
    axes = numpy.array(axes)
    n_axes = axes.size
    if filter_func == ndimage.rank_filter:
        args = (2,)
    elif filter_func == ndimage.percentile_filter:
        args = (30,)
    else:
        args = ()
    reduced_kwargs = {key: val[axes] for key, val in kwargs.items()}
    if len(set(axes % array.ndim)) != len(axes):
        with pytest.raises(ValueError, match='axes must be unique'):
            filter_func(array, *args, [size] * n_axes, axes=axes, **reduced_kwargs)
        return
    output = filter_func(array, *args, [size] * n_axes, axes=axes, **reduced_kwargs)
    size_3d = numpy.full(array.ndim, fill_value=size0)
    size_3d[axes] = size
    if 'origin' in kwargs:
        origin = numpy.array([0, 0, 0])
        origin[axes] = reduced_kwargs['origin']
        kwargs['origin'] = origin
    expected = filter_func(array, *args, size_3d, **kwargs)
    assert_allclose(output, expected)