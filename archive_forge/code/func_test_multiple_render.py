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
def test_multiple_render(self, df):
    s = Styler(df, uuid_len=0).map(lambda x: 'color: red;', subset=['A'])
    s.to_html()
    assert '<style type="text/css">\n#T__row0_col0, #T__row1_col0 {\n  color: red;\n}\n</style>' in s.to_html()