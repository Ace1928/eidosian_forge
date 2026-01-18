from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
def assert_ragged_arrays_equal(ra1, ra2):
    assert np.array_equal(ra1.start_indices, ra2.start_indices)
    assert np.array_equal(ra1.flat_array, ra2.flat_array)
    assert ra1.flat_array.dtype == ra2.flat_array.dtype
    for a1, a2 in zip(ra1, ra2):
        np.testing.assert_array_equal(a1, a2)