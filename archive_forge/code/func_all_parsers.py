from __future__ import annotations
import os
import pytest
from pandas.compat._optional import VERSIONS
from pandas import (
import pandas._testing as tm
@pytest.fixture(params=_all_parsers, ids=_all_parser_ids)
def all_parsers(request):
    """
    Fixture all of the CSV parsers.
    """
    parser = request.param()
    if parser.engine == 'pyarrow':
        pytest.importorskip('pyarrow', VERSIONS['pyarrow'])
        import pyarrow
        pyarrow.set_cpu_count(1)
    return parser