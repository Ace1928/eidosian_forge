import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_compare_complex_dtypes():
    df = pd.DataFrame(np.arange(5).astype(np.complex128))
    msg = "'<' not supported between instances of 'complex' and 'complex'"
    with pytest.raises(TypeError, match=msg):
        df < df.astype(object)
    with pytest.raises(TypeError, match=msg):
        df.lt(df.astype(object))