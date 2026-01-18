from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.compat import (
from pandas.compat.numpy import np_version_lt1p23
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.from_dataframe import from_dataframe
from pandas.core.interchange.utils import ArrowCTypes
def test_non_str_names_w_duplicates():
    df = pd.DataFrame({'0': [1, 2, 3], 0: [4, 5, 6]})
    dfi = df.__dataframe__()
    with pytest.raises(TypeError, match="Expected a Series, got a DataFrame. This likely happened because you called __dataframe__ on a DataFrame which, after converting column names to string, resulted in duplicated names: Index\\(\\['0', '0'\\], dtype='object'\\). Please rename these columns before using the interchange protocol."):
        pd.api.interchange.from_dataframe(dfi, allow_copy=False)