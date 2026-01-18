from collections.abc import Generator
from contextlib import contextmanager
import re
import struct
import tracemalloc
import numpy as np
import pytest
from pandas._libs import hashtable as ht
import pandas as pd
import pandas._testing as tm
from pandas.core.algorithms import isin
def test_add_different_nans(self):
    NAN1 = struct.unpack('d', struct.pack('=Q', 9221120237041090560))[0]
    NAN2 = struct.unpack('d', struct.pack('=Q', 9221120237041090561))[0]
    assert NAN1 != NAN1
    assert NAN2 != NAN2
    m = ht.Float64HashTable()
    m.set_item(NAN1, 0)
    m.set_item(NAN2, 0)
    assert len(m) == 1