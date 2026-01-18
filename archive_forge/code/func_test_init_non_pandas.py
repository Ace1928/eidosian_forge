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
def test_init_non_pandas(self):
    msg = '``data`` must be a Series or DataFrame'
    with pytest.raises(TypeError, match=msg):
        Styler([1, 2, 3])