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
@pytest.mark.parametrize('method', ['apply', 'map'])
@pytest.mark.parametrize('axis', ['index', 'columns'])
def test_apply_map_header_mi(mi_styler, method, axis):
    func = {'apply': lambda s: ['attr: val;' if 'b' in v else '' for v in s], 'map': lambda v: 'attr: val' if 'b' in v else ''}
    result = getattr(mi_styler, f'{method}_index')(func[method], axis=axis)._compute()
    expected = {(1, 1): [('attr', 'val')]}
    assert getattr(result, f'ctx_{axis}') == expected