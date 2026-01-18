from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
def test_construct_ragged_array_from_ragged_array():
    rarray = RaggedArray([[1, 2], [], [10, 20, 30], np.nan, [11, 22, 33, 44]], dtype='int32')
    result = RaggedArray(rarray)
    assert_ragged_arrays_equal(result, rarray)