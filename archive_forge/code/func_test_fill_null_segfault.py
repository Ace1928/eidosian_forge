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
def test_fill_null_segfault():
    arr = pa.array([None], pa.bool_()).fill_null(False)
    result = arr.cast(pa.int8())
    assert result == pa.array([0], pa.int8())