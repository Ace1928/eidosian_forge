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
def test_init_with_na_rep(self):
    df = DataFrame([[None, None], [1.1, 1.2]], columns=['A', 'B'])
    ctx = Styler(df, na_rep='NA')._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == 'NA'
    assert ctx['body'][0][2]['display_value'] == 'NA'