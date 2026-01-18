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
def test_hash_equal_tuple_with_nans():
    a = (float('nan'), (float('nan'), float('nan')))
    b = (float('nan'), (float('nan'), float('nan')))
    assert ht.object_hash(a) == ht.object_hash(b)
    assert ht.objects_are_equal(a, b)