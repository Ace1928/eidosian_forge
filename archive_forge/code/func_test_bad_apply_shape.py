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
def test_bad_apply_shape(self):
    df = DataFrame([[1, 2], [3, 4]], index=['A', 'B'], columns=['X', 'Y'])
    msg = 'resulted in the apply method collapsing to a Series.'
    with pytest.raises(ValueError, match=msg):
        df.style._apply(lambda x: 'x')
    msg = 'created invalid {} labels'
    with pytest.raises(ValueError, match=msg.format('index')):
        df.style._apply(lambda x: [''])
    with pytest.raises(ValueError, match=msg.format('index')):
        df.style._apply(lambda x: ['', '', '', ''])
    with pytest.raises(ValueError, match=msg.format('index')):
        df.style._apply(lambda x: Series(['a:v;', ''], index=['A', 'C']), axis=0)
    with pytest.raises(ValueError, match=msg.format('columns')):
        df.style._apply(lambda x: ['', '', ''], axis=1)
    with pytest.raises(ValueError, match=msg.format('columns')):
        df.style._apply(lambda x: Series(['a:v;', ''], index=['X', 'Z']), axis=1)
    msg = 'returned ndarray with wrong shape'
    with pytest.raises(ValueError, match=msg):
        df.style._apply(lambda x: np.array([[''], ['']]), axis=None)