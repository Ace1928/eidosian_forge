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
def test_non_reducing_slice_on_multiindex(self):
    dic = {('a', 'd'): [1, 4], ('a', 'c'): [2, 3], ('b', 'c'): [3, 2], ('b', 'd'): [4, 1]}
    df = DataFrame(dic, index=[0, 1])
    idx = IndexSlice
    slice_ = idx[:, idx['b', 'd']]
    tslice_ = non_reducing_slice(slice_)
    result = df.loc[tslice_]
    expected = DataFrame({('b', 'd'): [4, 1]})
    tm.assert_frame_equal(result, expected)