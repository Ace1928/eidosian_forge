from io import (
import os
import tempfile
import uuid
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_chunk_splits_multibyte_char(all_parsers):
    """
    Chunk splits a multibyte character with memory_map=True

    GH 43540
    """
    parser = all_parsers
    df = DataFrame(data=['a' * 127] * 2048)
    df.iloc[2047] = 'a' * 127 + 'ą'
    with tm.ensure_clean('bug-gh43540.csv') as fname:
        df.to_csv(fname, index=False, header=False, encoding='utf-8')
        if parser.engine == 'pyarrow':
            msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(fname, header=None, memory_map=True)
            return
        dfr = parser.read_csv(fname, header=None, memory_map=True)
    tm.assert_frame_equal(dfr, df)