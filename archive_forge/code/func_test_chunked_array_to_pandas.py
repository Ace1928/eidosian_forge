from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.pandas
def test_chunked_array_to_pandas():
    import pandas as pd
    data = [pa.array([-10, -5, 0, 5, 10])]
    table = pa.table(data, names=['a'])
    col = table.column(0)
    assert isinstance(col, pa.ChunkedArray)
    series = col.to_pandas()
    assert isinstance(series, pd.Series)
    assert series.shape == (5,)
    assert series[0] == -10
    assert series.name == 'a'