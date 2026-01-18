import codecs
import csv
from io import StringIO
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import PY311
from pandas.errors import (
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('nrows', [5, 3, None])
def test_malformed_chunks(all_parsers, nrows):
    data = 'ignore\nA,B,C\nskip\n1,2,3\n3,5,10 # comment\n1,2,3,4,5\n2,3,4\n'
    parser = all_parsers
    if parser.engine == 'pyarrow':
        msg = "The 'iterator' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), header=1, comment='#', iterator=True, chunksize=1, skiprows=[2])
        return
    msg = 'Expected 3 fields in line 6, saw 5'
    with parser.read_csv(StringIO(data), header=1, comment='#', iterator=True, chunksize=1, skiprows=[2]) as reader:
        with pytest.raises(ParserError, match=msg):
            reader.read(nrows)