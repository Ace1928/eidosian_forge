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
def check_partition_nth(data, indices, pivot, null_placement):
    indices = indices.to_pylist()
    assert len(indices) == len(data)
    assert sorted(indices) == list(range(len(data)))
    until_pivot = [data[indices[i]] for i in range(pivot)]
    after_pivot = [data[indices[i]] for i in range(pivot, len(data))]
    p = data[indices[pivot]]
    if p is None:
        if null_placement == 'at_start':
            assert all((v is None for v in until_pivot))
        else:
            assert all((v is None for v in after_pivot))
    elif null_placement == 'at_start':
        assert all((v is None or v <= p for v in until_pivot))
        assert all((v >= p for v in after_pivot))
    else:
        assert all((v <= p for v in until_pivot))
        assert all((v is None or v >= p for v in after_pivot))