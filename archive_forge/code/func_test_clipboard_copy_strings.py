from textwrap import dedent
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.clipboard import (
@pytest.mark.parametrize('sep', [None, 'default'])
def test_clipboard_copy_strings(self, sep, df):
    kwargs = build_kwargs(sep, False)
    df.to_clipboard(**kwargs)
    result = read_clipboard(sep='\\s+')
    assert result.to_string() == df.to_string()
    assert df.shape == result.shape