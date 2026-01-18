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
def test_drop_null_null_type():
    arr = pa.array([None] * 10)
    chunked_arr = pa.chunked_array([[None] * 5] * 2)
    batch = pa.record_batch([arr], names=['a'])
    table = pa.table({'a': arr})
    assert len(arr.drop_null()) == 0
    assert len(chunked_arr.drop_null()) == 0
    assert len(batch.drop_null().column(0)) == 0
    assert len(table.drop_null().column(0)) == 0