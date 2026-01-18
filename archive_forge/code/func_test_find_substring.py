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
def test_find_substring():
    for ty in [pa.string(), pa.binary(), pa.large_string(), pa.large_binary()]:
        arr = pa.array(['ab', 'cab', 'ba', None], type=ty)
        result = pc.find_substring(arr, 'ab')
        assert result.to_pylist() == [0, 1, -1, None]
        result = pc.find_substring_regex(arr, 'a?b')
        assert result.to_pylist() == [0, 1, 0, None]
        arr = pa.array(['ab*', 'cAB*', 'ba', 'aB?'], type=ty)
        result = pc.find_substring(arr, 'aB*', ignore_case=True)
        assert result.to_pylist() == [0, 1, -1, -1]
        result = pc.find_substring_regex(arr, 'a?b', ignore_case=True)
        assert result.to_pylist() == [0, 1, 0, 0]