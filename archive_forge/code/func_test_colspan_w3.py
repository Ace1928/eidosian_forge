from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_colspan_w3():
    df = DataFrame(data=[[1, 2]], columns=[['l0', 'l0'], ['l1a', 'l1b']])
    styler = Styler(df, uuid='_', cell_ids=False)
    assert '<th class="col_heading level0 col0" colspan="2">l0</th>' in styler.to_html()