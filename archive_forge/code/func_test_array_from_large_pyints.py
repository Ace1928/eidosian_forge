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
def test_array_from_large_pyints():
    with pytest.raises(OverflowError):
        pa.array([int(2 ** 63)])