import pytest
import decimal
import datetime
import pyarrow as pa
from pyarrow import fs
from pyarrow.tests import util
def test_bytesio_readwrite():
    from pyarrow import orc
    from io import BytesIO
    buf = BytesIO()
    a = pa.array([1, None, 3, None])
    b = pa.array([None, 'Arrow', None, 'ORC'])
    table = pa.table({'int64': a, 'utf8': b})
    orc.write_table(table, buf)
    buf.seek(0)
    orc_file = orc.ORCFile(buf)
    output_table = orc_file.read()
    assert table.equals(output_table)