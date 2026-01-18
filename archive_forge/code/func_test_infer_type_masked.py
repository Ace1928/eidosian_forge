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
def test_infer_type_masked():
    ty = pa.infer_type(['foo', 'bar', None, 2], mask=[False, False, False, True])
    assert ty == pa.utf8()
    ty = pa.infer_type(['foo', 'bar', None, 2], mask=np.array([True, True, True, True]))
    assert ty == pa.null()
    assert pa.infer_type([], mask=[]) == pa.null()