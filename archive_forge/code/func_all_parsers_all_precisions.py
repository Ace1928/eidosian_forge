from __future__ import annotations
import os
import pytest
from pandas.compat._optional import VERSIONS
from pandas import (
import pandas._testing as tm
@pytest.fixture(params=_get_all_parser_float_precision_combinations()['params'], ids=_get_all_parser_float_precision_combinations()['ids'])
def all_parsers_all_precisions(request):
    """
    Fixture for all allowable combinations of parser
    and float precision
    """
    return request.param