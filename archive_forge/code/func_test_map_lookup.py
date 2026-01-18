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
def test_map_lookup():
    ty = pa.map_(pa.utf8(), pa.int32())
    arr = pa.array([[('one', 1), ('two', 2)], [('none', 3)], [], [('one', 5), ('one', 7)], None], type=ty)
    result_first = pa.array([1, None, None, 5, None], type=pa.int32())
    result_last = pa.array([1, None, None, 7, None], type=pa.int32())
    result_all = pa.array([[1], None, None, [5, 7], None], type=pa.list_(pa.int32()))
    assert pc.map_lookup(arr, 'one', 'first') == result_first
    assert pc.map_lookup(arr, pa.scalar('one', type=pa.utf8()), 'first') == result_first
    assert pc.map_lookup(arr, pa.scalar('one', type=pa.utf8()), 'last') == result_last
    assert pc.map_lookup(arr, pa.scalar('one', type=pa.utf8()), 'all') == result_all