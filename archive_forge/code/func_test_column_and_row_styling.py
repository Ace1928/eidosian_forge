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
def test_column_and_row_styling(self):
    df = DataFrame(data=[[0, 1], [1, 2]], columns=['A', 'B'])
    s = Styler(df, uuid_len=0)
    s = s.set_table_styles({'A': [{'selector': '', 'props': [('color', 'blue')]}]})
    assert '#T_ .col0 {\n  color: blue;\n}' in s.to_html()
    s = s.set_table_styles({0: [{'selector': '', 'props': [('color', 'blue')]}]}, axis=1)
    assert '#T_ .row0 {\n  color: blue;\n}' in s.to_html()