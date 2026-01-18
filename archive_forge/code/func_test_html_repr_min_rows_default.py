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
def test_html_repr_min_rows_default(self, datapath):
    df = DataFrame({'a': range(20)})
    result = df._repr_html_()
    expected = expected_html(datapath, 'html_repr_min_rows_default_no_truncation')
    assert result == expected
    df = DataFrame({'a': range(61)})
    result = df._repr_html_()
    expected = expected_html(datapath, 'html_repr_min_rows_default_truncated')
    assert result == expected