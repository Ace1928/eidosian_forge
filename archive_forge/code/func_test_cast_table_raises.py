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
def test_cast_table_raises():
    table = pa.table({'a': [1, 2]})
    with pytest.raises(pa.lib.ArrowTypeError):
        pc.cast(table, pa.int64())