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
def test_arithmetic_add():
    left = pa.array([1, 2, 3, 4, 5])
    right = pa.array([0, -1, 1, 2, 3])
    result = pc.add(left, right)
    expected = pa.array([1, 1, 4, 6, 8])
    assert result.equals(expected)