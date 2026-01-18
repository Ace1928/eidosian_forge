import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
@parametrize_with_iterable_types
def test_nested_sequence_types(seq):
    arr1 = pa.array([seq([1, 2, 3])])
    arr2 = pa.array([[1, 2, 3]])
    assert arr1.equals(arr2)