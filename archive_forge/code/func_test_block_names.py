from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_block_names(tpl_style, tpl_table):
    expected_style = {'before_style', 'style', 'table_styles', 'before_cellstyle', 'cellstyle'}
    expected_table = {'before_table', 'table', 'caption', 'thead', 'tbody', 'after_table', 'before_head_rows', 'head_tr', 'after_head_rows', 'before_rows', 'tr', 'after_rows'}
    result1 = set(tpl_style.blocks)
    assert result1 == expected_style
    result2 = set(tpl_table.blocks)
    assert result2 == expected_table