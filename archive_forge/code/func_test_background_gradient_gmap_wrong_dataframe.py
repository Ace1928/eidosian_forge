import gc
import numpy as np
import pytest
from pandas import (
import matplotlib as mpl
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('gmap, axis', [(DataFrame([[1, 2], [2, 1]], columns=['A', 'B'], index=['X', 'Y']), 1), (DataFrame([[1, 2], [2, 1]], columns=['A', 'B'], index=['X', 'Y']), 0)])
def test_background_gradient_gmap_wrong_dataframe(styler_blank, gmap, axis):
    msg = "'gmap' is a DataFrame but underlying data for operations is a Series"
    with pytest.raises(ValueError, match=msg):
        styler_blank.background_gradient(gmap=gmap, axis=axis)._compute()