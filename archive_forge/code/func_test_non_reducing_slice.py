import contextlib
import copy
import re
from textwrap import dedent
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.formats.style import (  # isort:skip
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('slc', [IndexSlice[:, :], IndexSlice[:, 1], IndexSlice[1, :], IndexSlice[[1], [1]], IndexSlice[1, [1]], IndexSlice[[1], 1], IndexSlice[1], IndexSlice[1, 1], slice(None, None, None), [0, 1], np.array([0, 1]), Series([0, 1])])
def test_non_reducing_slice(self, slc):
    df = DataFrame([[0, 1], [2, 3]])
    tslice_ = non_reducing_slice(slc)
    assert isinstance(df.loc[tslice_], DataFrame)