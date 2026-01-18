from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
@pytest.mark.parametrize('start', (1.25, 10.5, -10.5))
@pytest.mark.parametrize('skip_nulls', (True, False))
def test_cumulative_prod(start, skip_nulls):
    start_int = int(start)
    starts = [None, start_int, pa.scalar(start_int, type=pa.int8()), pa.scalar(start_int, type=pa.int64())]
    for strt in starts:
        arrays = [pa.array([1, 2, 3]), pa.array([1, None, 20, 5]), pa.chunked_array([[1, None], [20, 5]])]
        expected_arrays = [pa.array([1, 2, 6]), pa.array([1, None, 20, 100]) if skip_nulls else pa.array([1, None, None, None]), pa.chunked_array([[1, None, 20, 100]]) if skip_nulls else pa.chunked_array([[1, None, None, None]])]
        for i, arr in enumerate(arrays):
            result = pc.cumulative_prod(arr, start=strt, skip_nulls=skip_nulls)
            expected = pc.multiply(expected_arrays[i], strt if strt is not None else 1)
            assert result.equals(expected)
    starts = [None, start, pa.scalar(start, type=pa.float32()), pa.scalar(start, type=pa.float64())]
    for strt in starts:
        arrays = [pa.array([1.5, 2.5, 3.5]), pa.array([1, np.nan, 2, -3, 4, 5]), pa.array([1, np.nan, None, 3, None, 5])]
        expected_arrays = [np.array([1.5, 3.75, 13.125]), np.array([1, np.nan, np.nan, np.nan, np.nan, np.nan]), np.array([1, np.nan, None, np.nan, None, np.nan]) if skip_nulls else np.array([1, np.nan, None, None, None, None])]
        for i, arr in enumerate(arrays):
            result = pc.cumulative_prod(arr, start=strt, skip_nulls=skip_nulls)
            expected = pc.multiply(expected_arrays[i], strt if strt is not None else 1)
            np.testing.assert_array_almost_equal(result.to_numpy(zero_copy_only=False), expected.to_numpy(zero_copy_only=False))
    for strt in ['a', pa.scalar('arrow'), 1.1]:
        with pytest.raises(pa.ArrowInvalid):
            pc.cumulative_prod([1, 2, 3], start=strt)