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
def test_call_function_with_memory_pool():
    arr = pa.array(['foo', 'bar', 'baz'])
    indices = np.array([2, 2, 1])
    result1 = arr.take(indices)
    result2 = pc.call_function('take', [arr, indices], memory_pool=pa.default_memory_pool())
    expected = pa.array(['baz', 'baz', 'bar'])
    assert result1.equals(expected)
    assert result2.equals(expected)
    result3 = pc.take(arr, indices, memory_pool=pa.default_memory_pool())
    assert result3.equals(expected)