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
def test_map_subset_multiindex_code(self):
    codes = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    columns = MultiIndex(levels=[['a', 'b'], ['%', '#']], codes=codes, names=['', ''])
    df = DataFrame([[1, -1, 1, 1], [-1, 1, 1, 1]], index=['hello', 'world'], columns=columns)
    pct_subset = IndexSlice[:, IndexSlice[:, '%':'%']]

    def color_negative_red(val):
        color = 'red' if val < 0 else 'black'
        return f'color: {color}'
    df.loc[pct_subset]
    df.style.map(color_negative_red, subset=pct_subset)