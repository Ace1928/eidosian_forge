import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
def test_no_reference_cycle(self):
    df = DataFrame({'a': [0, 1], 'b': [2, 3]})
    for name in ('loc', 'iloc', 'at', 'iat'):
        getattr(df, name)
    wr = weakref.ref(df)
    del df
    assert wr() is None