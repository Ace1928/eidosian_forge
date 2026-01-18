from io import StringIO
import os
from pathlib import Path
import pytest
from pandas.errors import ParserError
import pandas._testing as tm
from pandas.io.parsers import read_csv
import pandas.io.parsers.readers as parsers
@pytest.fixture(params=['python', 'python-fwf'], ids=lambda val: val)
def python_engine(request):
    return request.param