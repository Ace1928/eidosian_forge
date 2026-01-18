import gc
import numpy as np
import pytest
from pandas import (
import matplotlib as mpl
from pandas.io.formats.style import Styler
def test_background_gradient_int64():
    df1 = Series(range(3)).to_frame()
    df2 = Series(range(3), dtype='Int64').to_frame()
    ctx1 = df1.style.background_gradient()._compute().ctx
    ctx2 = df2.style.background_gradient()._compute().ctx
    assert ctx2[0, 0] == ctx1[0, 0]
    assert ctx2[1, 0] == ctx1[1, 0]
    assert ctx2[2, 0] == ctx1[2, 0]