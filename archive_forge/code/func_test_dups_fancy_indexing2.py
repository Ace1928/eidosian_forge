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
def test_dups_fancy_indexing2(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)), columns=['A', 'B', 'B', 'B', 'A'])
    with pytest.raises(KeyError, match='not in index'):
        df.loc[:, ['A', 'B', 'C']]