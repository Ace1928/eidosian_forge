import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@pytest.mark.parametrize('args, kwargs, expected', [([x, y], {}, [x, y, None, None]), ([x, y, triangles], {}, [x, y, triangles, None]), ([x, y], dict(triangles=triangles), [x, y, triangles, None]), ([x, y], dict(mask=mask), [x, y, None, mask]), ([x, y, triangles], dict(mask=mask), [x, y, triangles, mask]), ([x, y], dict(triangles=triangles, mask=mask), [x, y, triangles, mask])])
def test_extract_triangulation_params(self, args, kwargs, expected):
    other_args = [1, 2]
    other_kwargs = {'a': 3, 'b': '4'}
    x_, y_, triangles_, mask_, args_, kwargs_ = mtri.Triangulation._extract_triangulation_params(args + other_args, {**kwargs, **other_kwargs})
    x, y, triangles, mask = expected
    assert x_ is x
    assert y_ is y
    assert_array_equal(triangles_, triangles)
    assert mask_ is mask
    assert args_ == other_args
    assert kwargs_ == other_kwargs