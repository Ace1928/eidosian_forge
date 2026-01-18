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
@pytest.mark.parametrize('typ', ['array', 'chunked_array'])
def test_compare_string_scalar(typ):
    if typ == 'array':

        def con(values):
            return pa.array(values)
    else:

        def con(values):
            return pa.chunked_array([values])
    arr = con(['a', 'b', 'c', None])
    scalar = pa.scalar('b')
    result = pc.equal(arr, scalar)
    assert result.equals(con([False, True, False, None]))
    if typ == 'array':
        nascalar = pa.scalar(None, type='string')
        result = pc.equal(arr, nascalar)
        isnull = pc.is_null(result)
        assert isnull.equals(con([True, True, True, True]))
    result = pc.not_equal(arr, scalar)
    assert result.equals(con([True, False, True, None]))
    result = pc.less(arr, scalar)
    assert result.equals(con([True, False, False, None]))
    result = pc.less_equal(arr, scalar)
    assert result.equals(con([True, True, False, None]))
    result = pc.greater(arr, scalar)
    assert result.equals(con([False, False, True, None]))
    result = pc.greater_equal(arr, scalar)
    assert result.equals(con([False, True, True, None]))