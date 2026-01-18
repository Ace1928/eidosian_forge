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
def test_match_like():
    arr = pa.array(['ab', 'ba%', 'ba', 'ca%d', None])
    result = pc.match_like(arr, '_a\\%%')
    expected = pa.array([False, True, False, True, None])
    assert expected.equals(result)
    arr = pa.array(['aB', 'bA%', 'ba', 'ca%d', None])
    result = pc.match_like(arr, '_a\\%%', ignore_case=True)
    expected = pa.array([False, True, False, True, None])
    assert expected.equals(result)
    result = pc.match_like(arr, '_a\\%%', ignore_case=False)
    expected = pa.array([False, False, False, True, None])
    assert expected.equals(result)