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
def test_mode_array():
    arr = pa.array([1, 1, 3, 4, 3, 5], type='int64')
    mode = pc.mode(arr)
    assert len(mode) == 1
    assert mode[0].as_py() == {'mode': 1, 'count': 2}
    mode = pc.mode(arr, n=2)
    assert len(mode) == 2
    assert mode[0].as_py() == {'mode': 1, 'count': 2}
    assert mode[1].as_py() == {'mode': 3, 'count': 2}
    arr = pa.array([], type='int64')
    assert len(pc.mode(arr)) == 0
    arr = pa.array([1, 1, 3, 4, 3, None], type='int64')
    mode = pc.mode(arr, skip_nulls=False)
    assert len(mode) == 0
    mode = pc.mode(arr, min_count=6)
    assert len(mode) == 0
    mode = pc.mode(arr, skip_nulls=False, min_count=5)
    assert len(mode) == 0
    arr = pa.array([True, False])
    mode = pc.mode(arr, n=2)
    assert len(mode) == 2
    assert mode[0].as_py() == {'mode': False, 'count': 1}
    assert mode[1].as_py() == {'mode': True, 'count': 1}