from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('opname', ['skew', 'kurt', 'sem', 'prod', 'var'])
def test_invalid_td64_reductions(self, opname):
    s = Series([Timestamp('20130101') + timedelta(seconds=i * i) for i in range(10)])
    td = s.diff()
    msg = '|'.join([f"reduction operation '{opname}' not allowed for this dtype", f'cannot perform {opname} with type timedelta64\\[ns\\]', f"does not support reduction '{opname}'"])
    with pytest.raises(TypeError, match=msg):
        getattr(td, opname)()
    with pytest.raises(TypeError, match=msg):
        getattr(td.to_frame(), opname)(numeric_only=False)