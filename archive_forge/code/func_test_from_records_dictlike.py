from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_little_endian
from pandas import (
import pandas._testing as tm
def test_from_records_dictlike(self):
    df = DataFrame({'A': np.array(np.random.default_rng(2).standard_normal(6), dtype=np.float64), 'A1': np.array(np.random.default_rng(2).standard_normal(6), dtype=np.float64), 'B': np.array(np.arange(6), dtype=np.int64), 'C': ['foo'] * 6, 'D': np.array([True, False] * 3, dtype=bool), 'E': np.array(np.random.default_rng(2).standard_normal(6), dtype=np.float32), 'E1': np.array(np.random.default_rng(2).standard_normal(6), dtype=np.float32), 'F': np.array(np.arange(6), dtype=np.int32)})
    blocks = df._to_dict_of_blocks()
    columns = []
    for b in blocks.values():
        columns.extend(b.columns)
    asdict = dict(df.items())
    asdict2 = {x: y.values for x, y in df.items()}
    results = []
    results.append(DataFrame.from_records(asdict).reindex(columns=df.columns))
    results.append(DataFrame.from_records(asdict, columns=columns).reindex(columns=df.columns))
    results.append(DataFrame.from_records(asdict2, columns=columns).reindex(columns=df.columns))
    for r in results:
        tm.assert_frame_equal(r, df)