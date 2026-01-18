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
@pytest.mark.parametrize('method', ['map', 'apply'])
@pytest.mark.parametrize('axis', ['index', 'columns'])
def test_apply_map_header(method, axis):
    df = DataFrame({'A': [0, 0], 'B': [1, 1]}, index=['C', 'D'])
    func = {'apply': lambda s: ['attr: val' if 'A' in v or 'C' in v else '' for v in s], 'map': lambda v: 'attr: val' if 'A' in v or 'C' in v else ''}
    result = getattr(df.style, f'{method}_index')(func[method], axis=axis)
    assert len(result._todo) == 1
    assert len(getattr(result, f'ctx_{axis}')) == 0
    result._compute()
    expected = {(0, 0): [('attr', 'val')]}
    assert getattr(result, f'ctx_{axis}') == expected