from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_iloc_setitem_empty_frame_raises_with_3d_ndarray(self):
    idx = Index([])
    obj = DataFrame(np.random.default_rng(2).standard_normal((len(idx), len(idx))), index=idx, columns=idx)
    nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))
    msg = f'Cannot set values with ndim > {obj.ndim}'
    with pytest.raises(ValueError, match=msg):
        obj.iloc[nd3] = 0