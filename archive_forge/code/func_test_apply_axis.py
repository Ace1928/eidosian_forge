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
def test_apply_axis(self):
    df = DataFrame({'A': [0, 0], 'B': [1, 1]})
    f = lambda x: [f'val: {x.max()}' for v in x]
    result = df.style.apply(f, axis=1)
    assert len(result._todo) == 1
    assert len(result.ctx) == 0
    result._compute()
    expected = {(0, 0): [('val', '1')], (0, 1): [('val', '1')], (1, 0): [('val', '1')], (1, 1): [('val', '1')]}
    assert result.ctx == expected
    result = df.style.apply(f, axis=0)
    expected = {(0, 0): [('val', '0')], (0, 1): [('val', '1')], (1, 0): [('val', '0')], (1, 1): [('val', '1')]}
    result._compute()
    assert result.ctx == expected
    result = df.style.apply(f)
    result._compute()
    assert result.ctx == expected