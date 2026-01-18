import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
def test_hashing_struct_scalar():
    a = pa.array([[{'a': 5}, {'a': 6}], [{'a': 7}, None]])
    b = pa.array([[{'a': 7}, None]])
    hash1 = hash(a[1])
    hash2 = hash(b[0])
    assert hash1 == hash2