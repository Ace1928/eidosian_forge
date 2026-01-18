from io import StringIO
import os
from pathlib import Path
import pytest
from pandas.errors import ParserError
import pandas._testing as tm
from pandas.io.parsers import read_csv
import pandas.io.parsers.readers as parsers
def test_invalid_file_inputs(request, all_parsers):
    parser = all_parsers
    if parser.engine == 'python':
        request.applymarker(pytest.mark.xfail(reason=f'{parser.engine} engine supports lists.'))
    with pytest.raises(ValueError, match='Invalid'):
        parser.read_csv([])