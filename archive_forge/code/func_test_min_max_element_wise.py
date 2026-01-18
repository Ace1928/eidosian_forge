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
def test_min_max_element_wise():
    arr1 = pa.array([1, 2, 3])
    arr2 = pa.array([3, 1, 2])
    arr3 = pa.array([2, 3, None])
    result = pc.max_element_wise(arr1, arr2)
    assert result == pa.array([3, 2, 3])
    result = pc.min_element_wise(arr1, arr2)
    assert result == pa.array([1, 1, 2])
    result = pc.max_element_wise(arr1, arr2, arr3)
    assert result == pa.array([3, 3, 3])
    result = pc.min_element_wise(arr1, arr2, arr3)
    assert result == pa.array([1, 1, 2])
    result = pc.max_element_wise(arr1, arr3, skip_nulls=True)
    assert result == pa.array([2, 3, 3])
    result = pc.min_element_wise(arr1, arr3, skip_nulls=True)
    assert result == pa.array([1, 2, 3])
    result = pc.max_element_wise(arr1, arr3, options=pc.ElementWiseAggregateOptions())
    assert result == pa.array([2, 3, 3])
    result = pc.min_element_wise(arr1, arr3, options=pc.ElementWiseAggregateOptions())
    assert result == pa.array([1, 2, 3])
    result = pc.max_element_wise(arr1, arr3, skip_nulls=False)
    assert result == pa.array([2, 3, None])
    result = pc.min_element_wise(arr1, arr3, skip_nulls=False)
    assert result == pa.array([1, 2, None])