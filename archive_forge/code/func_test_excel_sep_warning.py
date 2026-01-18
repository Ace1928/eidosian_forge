from textwrap import dedent
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.clipboard import (
def test_excel_sep_warning(self, df):
    with tm.assert_produces_warning(UserWarning, match='to_clipboard in excel mode requires a single character separator.', check_stacklevel=False):
        df.to_clipboard(excel=True, sep='\\t')