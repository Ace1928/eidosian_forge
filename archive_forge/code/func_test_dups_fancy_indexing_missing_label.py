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
@pytest.mark.parametrize('vals', [[0, 1, 2], list('abc')])
def test_dups_fancy_indexing_missing_label(self, vals):
    df = DataFrame({'A': vals})
    with pytest.raises(KeyError, match='not in index'):
        df.loc[[0, 8, 0]]