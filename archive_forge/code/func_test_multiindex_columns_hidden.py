from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_multiindex_columns_hidden():
    df = DataFrame([[1, 2, 3, 4]])
    df.columns = MultiIndex.from_tuples([('A', 1), ('A', 2), ('A', 3), ('B', 1)])
    s = df.style
    assert '{tabular}{lrrrr}' in s.to_latex()
    s.set_table_styles([])
    s.hide([('A', 2)], axis='columns')
    assert '{tabular}{lrrr}' in s.to_latex()