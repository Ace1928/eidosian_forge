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
def test_match_substring_regex():
    arr = pa.array(['ab', 'abc', 'ba', 'c', None])
    result = pc.match_substring_regex(arr, '^a?b')
    expected = pa.array([True, True, True, False, None])
    assert expected.equals(result)
    arr = pa.array(['aB', 'Abc', 'BA', 'c', None])
    result = pc.match_substring_regex(arr, '^a?b', ignore_case=True)
    expected = pa.array([True, True, True, False, None])
    assert expected.equals(result)
    result = pc.match_substring_regex(arr, '^a?b', ignore_case=False)
    expected = pa.array([False, False, False, False, None])
    assert expected.equals(result)