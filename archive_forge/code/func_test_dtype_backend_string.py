from collections import defaultdict
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_dtype_backend_string(all_parsers, string_storage):
    pa = pytest.importorskip('pyarrow')
    with pd.option_context('mode.string_storage', string_storage):
        parser = all_parsers
        data = 'a,b\na,x\nb,\n'
        result = parser.read_csv(StringIO(data), dtype_backend='numpy_nullable')
        if string_storage == 'python':
            expected = DataFrame({'a': StringArray(np.array(['a', 'b'], dtype=np.object_)), 'b': StringArray(np.array(['x', pd.NA], dtype=np.object_))})
        else:
            expected = DataFrame({'a': ArrowStringArray(pa.array(['a', 'b'])), 'b': ArrowStringArray(pa.array(['x', None]))})
        tm.assert_frame_equal(result, expected)