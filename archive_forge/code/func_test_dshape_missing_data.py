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
def test_dshape_missing_data():
    assert discover([[1, 2, '', 3], [1, 2, '', 3], [1, 2, '', 3]]) == 3 * Tuple([int64, int64, null, int64])