from __future__ import annotations
import operator
from typing import (
import numpy as np
from pandas._libs import lib
from pandas._libs.missing import is_matching_na
from pandas._libs.sparse import SparseIndex
import pandas._libs.testing as _testing
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
from pandas import (
from pandas.core.arrays import (
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
from pandas.core.arrays.string_ import StringDtype
from pandas.core.indexes.api import safe_sort_index
from pandas.io.formats.printing import pprint_thing
def assert_is_sorted(seq) -> None:
    """Assert that the sequence is sorted."""
    if isinstance(seq, (Index, Series)):
        seq = seq.values
    if isinstance(seq, np.ndarray):
        assert_numpy_array_equal(seq, np.sort(np.array(seq)))
    else:
        assert_extension_array_equal(seq, seq[seq.argsort()])