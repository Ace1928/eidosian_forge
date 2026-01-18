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
@pytest.mark.parametrize('caption', [1, ('a', 'b', 'c'), (1, 's')])
def test_caption_raises(mi_styler, caption):
    msg = '`caption` must be either a string or 2-tuple of strings.'
    with pytest.raises(ValueError, match=msg):
        mi_styler.set_caption(caption)