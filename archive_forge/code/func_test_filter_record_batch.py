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
def test_filter_record_batch():
    batch = pa.record_batch([pa.array(['a', None, 'c', 'd', 'e'])], names=["a'"])
    mask = pa.array([True, False, None, False, True])
    result = batch.filter(mask)
    expected = pa.record_batch([pa.array(['a', 'e'])], names=["a'"])
    assert result.equals(expected)
    result = batch.filter(mask, null_selection_behavior='emit_null')
    expected = pa.record_batch([pa.array(['a', None, 'e'])], names=["a'"])
    assert result.equals(expected)