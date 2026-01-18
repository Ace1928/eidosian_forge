from io import StringIO
import os
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_dtype_chunksize_infer_categories(all_parsers):
    parser = all_parsers
    data = 'a,b\n1,a\n1,b\n1,b\n2,c'
    expecteds = [DataFrame({'a': [1, 1], 'b': Categorical(['a', 'b'])}), DataFrame({'a': [1, 2], 'b': Categorical(['b', 'c'])}, index=[2, 3])]
    if parser.engine == 'pyarrow':
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), dtype={'b': 'category'}, chunksize=2)
        return
    with parser.read_csv(StringIO(data), dtype={'b': 'category'}, chunksize=2) as actuals:
        for actual, expected in zip(actuals, expecteds):
            tm.assert_frame_equal(actual, expected)