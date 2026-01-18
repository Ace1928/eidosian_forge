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
@pytest.mark.xfail(raises=ValueError, reason='Not yet implemented')
def test_lowest_common_dshape_varlen_strings():
    assert lowest_common_dshape([String(10), String(11)]) == String(11)
    assert lowest_common_dshape([String(11), string]) == string