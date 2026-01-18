import datetime
import io
import warnings
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.pandas
def test_coerce_timestamps(tempdir):
    from collections import OrderedDict
    arrays = OrderedDict()
    fields = [pa.field('datetime64', pa.list_(pa.timestamp('ms')))]
    arrays['datetime64'] = [np.array(['2007-07-13T01:23:34.123456789', None, '2010-08-13T05:46:57.437699912'], dtype='datetime64[ms]'), None, None, np.array(['2007-07-13T02', None, '2010-08-13T05:46:57.437699912'], dtype='datetime64[ms]')]
    df = pd.DataFrame(arrays)
    schema = pa.schema(fields)
    filename = tempdir / 'pandas_roundtrip.parquet'
    arrow_table = pa.Table.from_pandas(df, schema=schema)
    _write_table(arrow_table, filename, version='2.6', coerce_timestamps='us')
    table_read = _read_table(filename)
    df_read = table_read.to_pandas()
    df_expected = df.copy()
    for i, x in enumerate(df_expected['datetime64']):
        if isinstance(x, np.ndarray):
            df_expected['datetime64'][i] = x.astype('M8[us]')
    tm.assert_frame_equal(df_expected, df_read)
    with pytest.raises(ValueError):
        _write_table(arrow_table, filename, version='2.6', coerce_timestamps='unknown')