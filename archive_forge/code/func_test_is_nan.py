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
def test_is_nan():
    arr = pa.array([1, 2, 3, None, np.nan])
    result = arr.is_nan()
    expected = pa.array([False, False, False, None, True])
    assert result.equals(expected)
    arr = pa.array(['1', '2', None], type=pa.string())
    with pytest.raises(ArrowNotImplementedError, match='has no kernel matching input types'):
        _ = arr.is_nan()
    with pytest.raises(ArrowNotImplementedError, match='has no kernel matching input types'):
        arr = pa.array([b'a', b'bb', None], type=pa.large_binary())
        _ = arr.is_nan()