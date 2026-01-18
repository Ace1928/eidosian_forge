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
def test_list_slice_bad_parameters():
    arr = pa.array([[1]], pa.list_(pa.int8(), 1))
    msg = '`start`(.*) should be greater than 0 and smaller than `stop`(.*)'
    with pytest.raises(pa.ArrowInvalid, match=msg):
        pc.list_slice(arr, -1, 1)
    with pytest.raises(pa.ArrowInvalid, match=msg):
        pc.list_slice(arr, 2, 1)
    with pytest.raises(pa.ArrowInvalid, match=msg):
        pc.list_slice(arr, 0, 0)
    msg = '`step` must be >= 1, got: '
    with pytest.raises(pa.ArrowInvalid, match=msg + '0'):
        pc.list_slice(arr, 0, 1, step=0)
    with pytest.raises(pa.ArrowInvalid, match=msg + '-1'):
        pc.list_slice(arr, 0, 1, step=-1)