from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
@pickle_test_parametrize
def test_array_pickle(data, typ, pickle_module):
    array = pa.array(data, type=typ)
    for proto in range(0, pickle_module.HIGHEST_PROTOCOL + 1):
        result = pickle_module.loads(pickle_module.dumps(array, proto))
        assert array.equals(result)