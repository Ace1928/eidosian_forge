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
def test_map_labelled():
    t = pa.map_(pa.field('name', 'string', nullable=False), 'int64')
    arr = pa.array([[('a', 1), ('b', 2)], [('c', 3)]], type=t)
    assert arr.type.key_field == pa.field('name', pa.utf8(), nullable=False)
    assert arr.type.item_field == pa.field('value', pa.int64())
    assert len(arr) == 2