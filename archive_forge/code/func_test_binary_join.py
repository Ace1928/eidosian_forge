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
def test_binary_join():
    ar_list = pa.array([['foo', 'bar'], None, []])
    expected = pa.array(['foo-bar', None, ''])
    assert pc.binary_join(ar_list, '-').equals(expected)
    separator_array = pa.array(['1', '2'], type=pa.binary())
    expected = pa.array(['a1b', 'c2d'], type=pa.binary())
    ar_list = pa.array([['a', 'b'], ['c', 'd']], type=pa.list_(pa.binary()))
    assert pc.binary_join(ar_list, separator_array).equals(expected)