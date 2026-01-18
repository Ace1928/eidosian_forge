from datetime import (
import itertools
import re
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.internals import (
from pandas.core.internals.blocks import (
@pytest.mark.parametrize('mgr_string, dtype', [('a: category', 'i8'), ('a: category; b: category', 'i8'), ('a: category; b: category2', 'object'), ('a: category2', 'object'), ('a: category2; b: category2', 'object'), ('a: f8', 'f8'), ('a: f8; b: i8', 'f8'), ('a: f4; b: i8', 'f8'), ('a: f4; b: i8; d: object', 'object'), ('a: bool; b: i8', 'object'), ('a: complex', 'complex'), ('a: f8; b: category', 'object'), ('a: M8[ns]; b: category', 'object'), ('a: M8[ns]; b: bool', 'object'), ('a: M8[ns]; b: i8', 'object'), ('a: m8[ns]; b: bool', 'object'), ('a: m8[ns]; b: i8', 'object'), ('a: M8[ns]; b: m8[ns]', 'object')])
def test_interleave_dtype(self, mgr_string, dtype):
    mgr = create_mgr('a: category')
    assert mgr.as_array().dtype == 'i8'
    mgr = create_mgr('a: category; b: category2')
    assert mgr.as_array().dtype == 'object'
    mgr = create_mgr('a: category2')
    assert mgr.as_array().dtype == 'object'
    mgr = create_mgr('a: f8')
    assert mgr.as_array().dtype == 'f8'
    mgr = create_mgr('a: f8; b: i8')
    assert mgr.as_array().dtype == 'f8'
    mgr = create_mgr('a: f4; b: i8')
    assert mgr.as_array().dtype == 'f8'
    mgr = create_mgr('a: f4; b: i8; d: object')
    assert mgr.as_array().dtype == 'object'
    mgr = create_mgr('a: bool; b: i8')
    assert mgr.as_array().dtype == 'object'
    mgr = create_mgr('a: complex')
    assert mgr.as_array().dtype == 'complex'
    mgr = create_mgr('a: f8; b: category')
    assert mgr.as_array().dtype == 'f8'
    mgr = create_mgr('a: M8[ns]; b: category')
    assert mgr.as_array().dtype == 'object'
    mgr = create_mgr('a: M8[ns]; b: bool')
    assert mgr.as_array().dtype == 'object'
    mgr = create_mgr('a: M8[ns]; b: i8')
    assert mgr.as_array().dtype == 'object'
    mgr = create_mgr('a: m8[ns]; b: bool')
    assert mgr.as_array().dtype == 'object'
    mgr = create_mgr('a: m8[ns]; b: i8')
    assert mgr.as_array().dtype == 'object'
    mgr = create_mgr('a: M8[ns]; b: m8[ns]')
    assert mgr.as_array().dtype == 'object'