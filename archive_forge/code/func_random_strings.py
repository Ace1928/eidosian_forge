import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
def random_strings(n, item_size, pct_null=0, dictionary=None):
    if dictionary is not None:
        result = dictionary[np.random.randint(0, len(dictionary), size=n)]
    else:
        result = np.array([random_ascii(item_size) for i in range(n)], dtype=object)
    if pct_null > 0:
        result[np.random.rand(n) < pct_null] = None
    return result