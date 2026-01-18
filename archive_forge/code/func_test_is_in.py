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
def test_is_in():
    arr = pa.array([1, 2, None, 1, 2, 3])
    result = pc.is_in(arr, value_set=pa.array([1, 3, None]))
    assert result.to_pylist() == [True, False, True, True, False, True]
    result = pc.is_in(arr, value_set=pa.array([1, 3, None]), skip_nulls=True)
    assert result.to_pylist() == [True, False, False, True, False, True]
    result = pc.is_in(arr, value_set=pa.array([1, 3]))
    assert result.to_pylist() == [True, False, False, True, False, True]
    result = pc.is_in(arr, value_set=pa.array([1, 3]), skip_nulls=True)
    assert result.to_pylist() == [True, False, False, True, False, True]