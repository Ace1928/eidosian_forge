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
def test_mi_sparse_column_names(self, blank_value):
    df = DataFrame(np.arange(16).reshape(4, 4), index=MultiIndex.from_arrays([['a', 'a', 'b', 'a'], [0, 1, 1, 2]], names=['idx_level_0', 'idx_level_1']), columns=MultiIndex.from_arrays([['C1', 'C1', 'C2', 'C2'], [1, 0, 1, 0]], names=['colnam_0', 'colnam_1']))
    result = Styler(df, cell_ids=False)._translate(True, True)
    for level in [0, 1]:
        head = result['head'][level]
        expected = [{'class': 'blank', 'display_value': blank_value, 'is_visible': True}, {'class': f'index_name level{level}', 'display_value': f'colnam_{level}', 'is_visible': True}]
        for i, expected_dict in enumerate(expected):
            assert expected_dict.items() <= head[i].items()