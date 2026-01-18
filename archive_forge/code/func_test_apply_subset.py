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
@pytest.mark.parametrize('axis', [0, 1])
def test_apply_subset(self, slice_, axis, df):

    def h(x, color='bar'):
        return Series(f'color: {color}', index=x.index, name=x.name)
    result = df.style.apply(h, axis=axis, subset=slice_, color='baz')._compute().ctx
    expected = {(r, c): [('color', 'baz')] for r, row in enumerate(df.index) for c, col in enumerate(df.columns) if row in df.loc[slice_].index and col in df.loc[slice_].columns}
    assert result == expected