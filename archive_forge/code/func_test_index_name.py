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
def test_index_name(self):
    df = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    result = df.set_index('A').style._translate(True, True)
    expected = {'class': 'index_name level0', 'type': 'th', 'value': 'A', 'is_visible': True, 'display_value': 'A'}
    assert expected.items() <= result['head'][1][0].items()