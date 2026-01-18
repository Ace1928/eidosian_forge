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
def test_list_of_dicts_no_difference():
    data = [{'name': 'Alice', 'amount': 100}, {'name': 'Bob'}]
    result = discover(data)
    expected = dshape('2 * {amount: ?int64, name: string}')
    assert result == expected