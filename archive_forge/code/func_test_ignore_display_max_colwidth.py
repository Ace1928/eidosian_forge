from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
@pytest.mark.parametrize('method,expected', [('to_html', lambda x: lorem_ipsum), ('_repr_html_', lambda x: lorem_ipsum[:x - 4] + '...')])
@pytest.mark.parametrize('max_colwidth', [10, 20, 50, 100])
def test_ignore_display_max_colwidth(method, expected, max_colwidth):
    df = DataFrame([lorem_ipsum])
    with option_context('display.max_colwidth', max_colwidth):
        result = getattr(df, method)()
    expected = expected(max_colwidth)
    assert expected in result