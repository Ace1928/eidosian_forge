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
@pytest.mark.parametrize('start', (0.5, 3.5, 6.5))
@pytest.mark.parametrize('skip_nulls', (True, False))
def test_cumulative_max(start, skip_nulls):
    start_int = int(start)
    starts = [None, start_int, pa.scalar(start_int, type=pa.int8()), pa.scalar(start_int, type=pa.int64())]
    for strt in starts:
        arrays = [pa.array([2, 1, 3, 5, 4, 6]), pa.array([2, 1, None, 5, 4, None]), pa.chunked_array([[2, 1, None], [5, 4, None]])]
        expected_arrays = [pa.array([2, 2, 3, 5, 5, 6]), pa.array([2, 2, None, 5, 5, None]) if skip_nulls else pa.array([2, 2, None, None, None, None]), pa.chunked_array([[2, 2, None, 5, 5, None]]) if skip_nulls else pa.chunked_array([[2, 2, None, None, None, None]])]
        for i, arr in enumerate(arrays):
            result = pc.cumulative_max(arr, start=strt, skip_nulls=skip_nulls)
            expected = pc.max_element_wise(expected_arrays[i], strt if strt is not None else int(-1000000000.0), skip_nulls=False)
            assert result.equals(expected)
    starts = [None, start, pa.scalar(start, type=pa.float32()), pa.scalar(start, type=pa.float64())]
    for strt in starts:
        arrays = [pa.array([2.5, 1.3, 3.7, 5.1, 4.9, 6.2]), pa.array([2.5, 1.3, 3.7, np.nan, 4.9, 6.2]), pa.array([2.5, 1.3, None, np.nan, 4.9, None])]
        expected_arrays = [np.array([2.5, 2.5, 3.7, 5.1, 5.1, 6.2]), np.array([2.5, 2.5, 3.7, 3.7, 4.9, 6.2]), np.array([2.5, 2.5, None, 2.5, 4.9, None]) if skip_nulls else np.array([2.5, 2.5, None, None, None, None])]
        for i, arr in enumerate(arrays):
            result = pc.cumulative_max(arr, start=strt, skip_nulls=skip_nulls)
            expected = pc.max_element_wise(expected_arrays[i], strt if strt is not None else -1000000000.0, skip_nulls=False)
            np.testing.assert_array_almost_equal(result.to_numpy(zero_copy_only=False), expected.to_numpy(zero_copy_only=False))
    for strt in ['a', pa.scalar('arrow'), 1.1]:
        with pytest.raises(pa.ArrowInvalid):
            pc.cumulative_max([1, 2, 3], start=strt)