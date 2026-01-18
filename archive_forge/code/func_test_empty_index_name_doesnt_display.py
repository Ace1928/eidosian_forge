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
def test_empty_index_name_doesnt_display(self, blank_value):
    df = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    result = df.style._translate(True, True)
    assert len(result['head']) == 1
    expected = {'class': 'blank level0', 'type': 'th', 'value': blank_value, 'is_visible': True, 'display_value': blank_value}
    assert expected.items() <= result['head'][0][0].items()