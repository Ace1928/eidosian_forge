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
@pytest.mark.parametrize(('ty', 'values'), all_array_types)
def test_drop_null(ty, values):
    arr = pa.array(values, type=ty)
    result = arr.drop_null()
    result.validate(full=True)
    indices = [i for i in range(len(arr)) if arr[i].is_valid]
    expected = arr.take(pa.array(indices))
    assert result.equals(expected)