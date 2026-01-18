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
@pytest.mark.parametrize('slice_', [IndexSlice[:, :], IndexSlice[:, IndexSlice[['a']]], IndexSlice[:, IndexSlice[['a'], ['c']]], IndexSlice[:, IndexSlice['a', 'c', :]], IndexSlice[:, IndexSlice['a', :, 'e']], IndexSlice[:, IndexSlice[:, 'c', 'e']], IndexSlice[:, IndexSlice['a', ['c', 'd'], :]], IndexSlice[:, IndexSlice['a', ['c', 'd', '-'], :]], IndexSlice[:, IndexSlice['a', ['c', 'd', '-'], 'e']], IndexSlice[IndexSlice[['U']], :], IndexSlice[IndexSlice[['U'], ['W']], :], IndexSlice[IndexSlice['U', 'W', :], :], IndexSlice[IndexSlice['U', :, 'Y'], :], IndexSlice[IndexSlice[:, 'W', 'Y'], :], IndexSlice[IndexSlice[:, 'W', ['Y', 'Z']], :], IndexSlice[IndexSlice[:, 'W', ['Y', 'Z', '-']], :], IndexSlice[IndexSlice['U', 'W', ['Y', 'Z', '-']], :], IndexSlice[IndexSlice[:, 'W', 'Y'], IndexSlice['a', 'c', :]]])
def test_non_reducing_multi_slice_on_multiindex(self, slice_):
    cols = MultiIndex.from_product([['a', 'b'], ['c', 'd'], ['e', 'f']])
    idxs = MultiIndex.from_product([['U', 'V'], ['W', 'X'], ['Y', 'Z']])
    df = DataFrame(np.arange(64).reshape(8, 8), columns=cols, index=idxs)
    for lvl in [0, 1]:
        key = slice_[lvl]
        if isinstance(key, tuple):
            for subkey in key:
                if isinstance(subkey, list) and '-' in subkey:
                    with pytest.raises(KeyError, match='-'):
                        df.loc[slice_]
                    return
    expected = df.loc[slice_]
    result = df.loc[non_reducing_slice(slice_)]
    tm.assert_frame_equal(result, expected)