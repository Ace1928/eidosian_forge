import gc
import numpy as np
import pytest
from pandas import (
import matplotlib as mpl
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('gmap, axis, exp_gmap', [(Series([2, 1], index=['Y', 'X']), 0, [[1, 1], [2, 2]]), (Series([2, 1], index=['B', 'A']), 1, [[1, 2], [1, 2]]), (Series([1, 2, 3], index=['X', 'Y', 'Z']), 0, [[1, 1], [2, 2]]), (Series([1, 2, 3], index=['A', 'B', 'C']), 1, [[1, 2], [1, 2]])])
def test_background_gradient_gmap_series_align(styler_blank, gmap, axis, exp_gmap):
    expected = styler_blank.background_gradient(axis=None, gmap=exp_gmap)._compute()
    result = styler_blank.background_gradient(axis=axis, gmap=gmap)._compute()
    assert expected.ctx == result.ctx