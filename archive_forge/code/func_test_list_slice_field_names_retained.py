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
@pytest.mark.parametrize('return_fixed_size', (True, False, None))
@pytest.mark.parametrize('type', (lambda: pa.list_(pa.field('col', pa.int8())), lambda: pa.list_(pa.field('col', pa.int8()), 1), lambda: pa.large_list(pa.field('col', pa.int8()))))
def test_list_slice_field_names_retained(return_fixed_size, type):
    arr = pa.array([[1]], type())
    out = pc.list_slice(arr, 0, 1, return_fixed_size_list=return_fixed_size)
    assert arr.type.field(0).name == out.type.field(0).name
    if return_fixed_size is None:
        assert arr.type == out.type