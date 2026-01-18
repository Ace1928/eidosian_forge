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
def test_get_level_lengths(self):
    index = MultiIndex.from_product([['a', 'b'], [0, 1, 2]])
    expected = {(0, 0): 3, (0, 3): 3, (1, 0): 1, (1, 1): 1, (1, 2): 1, (1, 3): 1, (1, 4): 1, (1, 5): 1}
    result = _get_level_lengths(index, sparsify=True, max_index=100)
    tm.assert_dict_equal(result, expected)
    expected = {(0, 0): 1, (0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 4): 1, (0, 5): 1, (1, 0): 1, (1, 1): 1, (1, 2): 1, (1, 3): 1, (1, 4): 1, (1, 5): 1}
    result = _get_level_lengths(index, sparsify=False, max_index=100)
    tm.assert_dict_equal(result, expected)