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
def test_drop_null_chunked_array():
    arr = pa.chunked_array([['a', None], ['c', 'd', None], [None], []])
    expected_drop = pa.chunked_array([['a'], ['c', 'd'], [], []])
    result = arr.drop_null()
    assert result.equals(expected_drop)