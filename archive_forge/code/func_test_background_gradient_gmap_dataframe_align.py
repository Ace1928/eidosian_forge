import gc
import numpy as np
import pytest
from pandas import (
import matplotlib as mpl
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('gmap', [DataFrame([[2, 1], [1, 2]], columns=['B', 'A'], index=['X', 'Y']), DataFrame([[2, 1], [1, 2]], columns=['A', 'B'], index=['Y', 'X']), DataFrame([[1, 2], [2, 1]], columns=['B', 'A'], index=['Y', 'X']), DataFrame([[1, 2, 3], [2, 1, 3]], columns=['A', 'B', 'C'], index=['X', 'Y']), DataFrame([[1, 2], [2, 1], [3, 3]], columns=['A', 'B'], index=['X', 'Y', 'Z'])])
@pytest.mark.parametrize('subset, exp_gmap', [(None, [[1, 2], [2, 1]]), (['A'], [[1], [2]]), (['B', 'A'], [[2, 1], [1, 2]]), (IndexSlice['X', :], [[1, 2]]), (IndexSlice[['Y', 'X'], :], [[2, 1], [1, 2]])])
def test_background_gradient_gmap_dataframe_align(styler_blank, gmap, subset, exp_gmap):
    expected = styler_blank.background_gradient(axis=None, gmap=exp_gmap, subset=subset)
    result = styler_blank.background_gradient(axis=None, gmap=gmap, subset=subset)
    assert expected._compute().ctx == result._compute().ctx