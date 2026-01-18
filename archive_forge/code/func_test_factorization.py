from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
def test_factorization():
    arg = np.array([[1, 2], [], [1, 2], None, [11, 22, 33, 44]], dtype=object)
    rarray = RaggedArray(arg, dtype='int16')
    labels, uniques = rarray.factorize()
    np.testing.assert_array_equal(labels, [0, -1, 0, -1, 1])
    assert_ragged_arrays_equal(uniques, RaggedArray([[1, 2], [11, 22, 33, 44]], dtype='int16'))