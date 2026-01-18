import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('dtype, rdtype', dtypes + [('object', int), ('category', int)])
def test_iterable_map(self, index_or_series, dtype, rdtype):
    typ = index_or_series
    if dtype == 'float16' and issubclass(typ, pd.Index):
        with pytest.raises(NotImplementedError, match='float16 indexes are not '):
            typ([1], dtype=dtype)
        return
    s = typ([1], dtype=dtype)
    result = s.map(type)[0]
    if not isinstance(rdtype, tuple):
        rdtype = (rdtype,)
    assert result in rdtype