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
@pytest.mark.parametrize('sparse_index, exp_rows', [(True, [{'is_visible': True, 'attributes': 'rowspan="2"', 'value': 'i0'}, {'is_visible': False, 'attributes': '', 'value': 'i0'}]), (False, [{'is_visible': True, 'attributes': '', 'value': 'i0'}, {'is_visible': True, 'attributes': '', 'value': 'i0'}])])
def test_mi_styler_sparsify_index(mi_styler, sparse_index, exp_rows):
    exp_l1_r0 = {'is_visible': True, 'attributes': '', 'display_value': 'i1_a'}
    exp_l1_r1 = {'is_visible': True, 'attributes': '', 'display_value': 'i1_b'}
    ctx = mi_styler._translate(sparse_index, True)
    assert exp_rows[0].items() <= ctx['body'][0][0].items()
    assert exp_rows[1].items() <= ctx['body'][1][0].items()
    assert exp_l1_r0.items() <= ctx['body'][0][1].items()
    assert exp_l1_r1.items() <= ctx['body'][1][1].items()