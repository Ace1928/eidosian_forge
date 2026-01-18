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
@pytest.mark.parametrize('slice_', [IndexSlice[:, IndexSlice['x', 'A']], IndexSlice[:, IndexSlice[:, 'A']], IndexSlice[:, IndexSlice[:, ['A', 'C']]], IndexSlice[IndexSlice['a', 1], :], IndexSlice[IndexSlice[:, 1], :], IndexSlice[IndexSlice[:, [1, 3]], :], IndexSlice[:, ('x', 'A')], IndexSlice[('a', 1), :]])
def test_map_subset_multiindex(self, slice_):
    if isinstance(slice_[-1], tuple) and isinstance(slice_[-1][-1], list) and ('C' in slice_[-1][-1]):
        ctx = pytest.raises(KeyError, match='C')
    elif isinstance(slice_[0], tuple) and isinstance(slice_[0][1], list) and (3 in slice_[0][1]):
        ctx = pytest.raises(KeyError, match='3')
    else:
        ctx = contextlib.nullcontext()
    idx = MultiIndex.from_product([['a', 'b'], [1, 2]])
    col = MultiIndex.from_product([['x', 'y'], ['A', 'B']])
    df = DataFrame(np.random.default_rng(2).random((4, 4)), columns=col, index=idx)
    with ctx:
        df.style.map(lambda x: 'color: red;', subset=slice_).to_html()