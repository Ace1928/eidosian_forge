import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
@pytest.mark.pandas
def test_list_from_pandas():
    import pandas as pd
    s = pa.scalar(pd.Series([1, 2, 3]))
    assert s.as_py() == [1, 2, 3]
    cases = [(np.nan, 'null'), (['string', np.nan], pa.list_(pa.binary())), (['string', np.nan], pa.list_(pa.utf8())), ([b'string', np.nan], pa.list_(pa.binary(6))), ([True, np.nan], pa.list_(pa.bool_())), ([decimal.Decimal('0'), np.nan], pa.list_(pa.decimal128(12, 2)))]
    for case, ty in cases:
        with pytest.raises((ValueError, TypeError)):
            pa.scalar(case, type=ty)
        s = pa.scalar(case, type=ty, from_pandas=True)