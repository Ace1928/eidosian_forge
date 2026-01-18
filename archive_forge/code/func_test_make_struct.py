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
def test_make_struct():
    assert pc.make_struct(1, 'a').as_py() == {'0': 1, '1': 'a'}
    assert pc.make_struct(1, 'a', field_names=['i', 's']).as_py() == {'i': 1, 's': 'a'}
    assert pc.make_struct([1, 2, 3], 'a b c'.split()) == pa.StructArray.from_arrays([[1, 2, 3], 'a b c'.split()], names='0 1'.split())
    with pytest.raises(ValueError, match='Array arguments must all be the same length'):
        pc.make_struct([1, 2, 3, 4], 'a b c'.split())
    with pytest.raises(ValueError, match='0 arguments but 2 field names'):
        pc.make_struct(field_names=['one', 'two'])