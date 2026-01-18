from __future__ import annotations
import os
import pytest
from pandas.compat._optional import VERSIONS
from pandas import (
import pandas._testing as tm
@pytest.fixture
def pyarrow_xfail(request):
    """
    Fixture that xfails a test if the engine is pyarrow.

    Use if failure is do to unsupported keywords or inconsistent results.
    """
    if 'all_parsers' in request.fixturenames:
        parser = request.getfixturevalue('all_parsers')
    elif 'all_parsers_all_precisions' in request.fixturenames:
        parser = request.getfixturevalue('all_parsers_all_precisions')[0]
    else:
        return
    if parser.engine == 'pyarrow':
        mark = pytest.mark.xfail(reason="pyarrow doesn't support this.")
        request.applymarker(mark)