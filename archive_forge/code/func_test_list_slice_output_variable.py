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
@pytest.mark.parametrize('start,stop', ((0, None), (0, 1), (0, 2), (1, 2), (2, 4)))
@pytest.mark.parametrize('step', (1, 2))
@pytest.mark.parametrize('value_type', (pa.string, pa.int16, pa.float64))
@pytest.mark.parametrize('list_type', (pa.list_, pa.large_list, 'fixed'))
def test_list_slice_output_variable(start, stop, step, value_type, list_type):
    if list_type == 'fixed':
        data = [[1, 2, 3], [4, 5, None], [6, None, None], None]
        arr = pa.array(data, pa.list_(pa.int8(), 3)).cast(pa.list_(value_type(), 3))
    else:
        data = [[1, 2, 3], [4, 5], [6], None]
        arr = pa.array(data, pa.list_(pa.int8())).cast(list_type(value_type()))
    if list_type == 'fixed':
        list_type = pa.list_
    result = pc.list_slice(arr, start, stop, step, return_fixed_size_list=False)
    assert result.type == list_type(value_type())
    pylist = result.cast(pa.list_(pa.int8())).to_pylist()
    expected = [d[start:stop:step] if d is not None else None for d in data]
    assert pylist == expected