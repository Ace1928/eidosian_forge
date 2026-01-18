from io import (
import os
import platform
from urllib.error import URLError
import uuid
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_context_manageri_user_provided(all_parsers, datapath):
    parser = all_parsers
    with open(datapath('io', 'data', 'csv', 'iris.csv'), encoding='utf-8') as path:
        if parser.engine == 'pyarrow':
            msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(path, chunksize=1)
            return
        reader = parser.read_csv(path, chunksize=1)
        assert not reader.handles.handle.closed
        try:
            with reader:
                next(reader)
                assert False
        except AssertionError:
            assert not reader.handles.handle.closed