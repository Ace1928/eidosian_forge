import contextlib
from datetime import datetime
import io
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import EmptyDataError
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sas7bdat import SAS7BDATReader
@pytest.mark.slow
@pytest.mark.parametrize('chunksize', (3, 5, 10, 11))
@pytest.mark.parametrize('k', range(1, 17))
def test_iterator_loop(self, dirpath, k, chunksize):
    fname = os.path.join(dirpath, f'test{k}.sas7bdat')
    with pd.read_sas(fname, chunksize=chunksize, encoding='utf-8') as rdr:
        y = 0
        for x in rdr:
            y += x.shape[0]
    assert y == rdr.row_count