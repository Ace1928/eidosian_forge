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
def test_array_factory_invalid_type():

    class MyObject:
        pass
    arr = np.array([MyObject()])
    with pytest.raises(ValueError):
        pa.array(arr)