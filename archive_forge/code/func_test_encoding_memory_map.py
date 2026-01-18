from io import (
import os
import tempfile
import uuid
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('encoding', ['utf-8', None, 'utf-16', 'cp1255', 'latin-1'])
def test_encoding_memory_map(all_parsers, encoding):
    parser = all_parsers
    expected = DataFrame({'name': ['Raphael', 'Donatello', 'Miguel Angel', 'Leonardo'], 'mask': ['red', 'purple', 'orange', 'blue'], 'weapon': ['sai', 'bo staff', 'nunchunk', 'katana']})
    with tm.ensure_clean() as file:
        expected.to_csv(file, index=False, encoding=encoding)
        if parser.engine == 'pyarrow':
            msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(file, encoding=encoding, memory_map=True)
            return
        df = parser.read_csv(file, encoding=encoding, memory_map=True)
    tm.assert_frame_equal(df, expected)