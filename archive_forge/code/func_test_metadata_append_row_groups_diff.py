import datetime
import decimal
from collections import OrderedDict
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip, make_sample_file
from pyarrow.fs import LocalFileSystem
from pyarrow.tests import util
@pytest.mark.parametrize('t1,t2,expected_error', (({'col1': range(10)}, {'col1': range(10)}, None), ({'col1': range(10)}, {'col2': range(10)}, 'The two columns with index 0 differ.'), ({'col1': range(10), 'col2': range(10)}, {'col3': range(10)}, 'This schema has 2 columns, other has 1')))
def test_metadata_append_row_groups_diff(t1, t2, expected_error):
    table1 = pa.table(t1)
    table2 = pa.table(t2)
    buf1 = io.BytesIO()
    buf2 = io.BytesIO()
    pq.write_table(table1, buf1)
    pq.write_table(table2, buf2)
    buf1.seek(0)
    buf2.seek(0)
    meta1 = pq.ParquetFile(buf1).metadata
    meta2 = pq.ParquetFile(buf2).metadata
    if expected_error:
        prefix = 'AppendRowGroups requires equal schemas.\n'
        with pytest.raises(RuntimeError, match=prefix + expected_error):
            meta1.append_row_groups(meta2)
    else:
        meta1.append_row_groups(meta2)