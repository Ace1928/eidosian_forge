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
def test_filter_chunked_array():
    arr = pa.chunked_array([['a', None], ['c', 'd', 'e']])
    expected_drop = pa.chunked_array([['a'], ['e']])
    expected_null = pa.chunked_array([['a'], [None, 'e']])
    for mask in [pa.array([True, False, None, False, True]), pa.chunked_array([[True, False, None], [False, True]]), [True, False, None, False, True]]:
        result = arr.filter(mask)
        assert result.equals(expected_drop)
        result = arr.filter(mask, null_selection_behavior='emit_null')
        assert result.equals(expected_null)