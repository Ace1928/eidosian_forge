from collections import OrderedDict
from itertools import starmap
from types import MappingProxyType
from warnings import catch_warnings, simplefilter
import numpy as np
import pytest
from datashader.datashape.discovery import (
from datashader.datashape.coretypes import (
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape import dshape
from datetime import date, time, datetime, timedelta
def test_discover_mixed():
    i = discover(1)
    f = discover(1.0)
    exp = 10 * Tuple([i, i, f, f])
    assert dshape(discover([[1, 2, 1.0, 2.0]] * 10)) == exp
    exp = 10 * (4 * f)
    assert dshape(discover([[1, 2, 1.0, 2.0], [1.0, 2.0, 1, 2]] * 5)) == exp