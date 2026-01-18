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
def test_multi_chunk_pyarrow() -> None:
    pa = pytest.importorskip('pyarrow', '11.0.0')
    n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
    names = ['n_legs']
    table = pa.table([n_legs], names=names)
    with pytest.raises(RuntimeError, match='To join chunks a copy is required which is forbidden by allow_copy=False'):
        pd.api.interchange.from_dataframe(table, allow_copy=False)