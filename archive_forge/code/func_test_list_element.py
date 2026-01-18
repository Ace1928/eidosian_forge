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
def test_list_element():
    element_type = pa.struct([('a', pa.float64()), ('b', pa.int8())])
    list_type = pa.list_(element_type)
    l1 = [{'a': 0.4, 'b': 2}, None, {'a': 0.2, 'b': 4}, None, {'a': 5.6, 'b': 6}]
    l2 = [None, {'a': 0.52, 'b': 3}, {'a': 0.7, 'b': 4}, None, {'a': 0.6, 'b': 8}]
    lists = pa.array([l1, l2], list_type)
    index = 1
    result = pa.compute.list_element(lists, index)
    expected = pa.array([None, {'a': 0.52, 'b': 3}], element_type)
    assert result.equals(expected)
    index = 4
    result = pa.compute.list_element(lists, index)
    expected = pa.array([{'a': 5.6, 'b': 6}, {'a': 0.6, 'b': 8}], element_type)
    assert result.equals(expected)