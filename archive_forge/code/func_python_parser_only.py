from __future__ import annotations
import os
import pytest
from pandas.compat._optional import VERSIONS
from pandas import (
import pandas._testing as tm
@pytest.fixture(params=_py_parsers_only, ids=_py_parser_ids)
def python_parser_only(request):
    """
    Fixture all of the CSV parsers using the Python engine.
    """
    return request.param()