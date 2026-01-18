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
def test_caption(self, df):
    styler = Styler(df, caption='foo')
    result = styler.to_html()
    assert all(['caption' in result, 'foo' in result])
    styler = df.style
    result = styler.set_caption('baz')
    assert styler is result
    assert styler.caption == 'baz'