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
@pytest.mark.parametrize('slice_', [IndexSlice[:], IndexSlice[:, ['A']], IndexSlice[[1], :], IndexSlice[[1], ['A']], IndexSlice[:2, ['A', 'B']]])
def test_map_subset(self, slice_, df):
    result = df.style.map(lambda x: 'color:baz;', subset=slice_)._compute().ctx
    expected = {(r, c): [('color', 'baz')] for r, row in enumerate(df.index) for c, col in enumerate(df.columns) if row in df.loc[slice_].index and col in df.loc[slice_].columns}
    assert result == expected