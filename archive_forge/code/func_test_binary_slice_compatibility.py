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
def test_binary_slice_compatibility():
    arr = pa.array([b'', b'a', b'a\xff', b'ab\x00', b'abc\xfb', b'ab\xf2de'])
    for start, stop, step in itertools.product(range(-6, 6), range(-6, 6), range(-3, 4)):
        if step == 0:
            continue
        expected = pa.array([k.as_py()[start:stop:step] for k in arr])
        result = pc.binary_slice(arr, start=start, stop=stop, step=step)
        assert expected.equals(result)
        assert pc.binary_slice(arr, start, stop, step) == result