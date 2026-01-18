import datetime
import io
import warnings
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.pandas
def test_coerce_timestamps_truncated(tempdir):
    """
    ARROW-2555: Test that we can truncate timestamps when coercing if
    explicitly allowed.
    """
    dt_us = datetime.datetime(year=2017, month=1, day=1, hour=1, minute=1, second=1, microsecond=1)
    dt_ms = datetime.datetime(year=2017, month=1, day=1, hour=1, minute=1, second=1)
    fields_us = [pa.field('datetime64', pa.timestamp('us'))]
    arrays_us = {'datetime64': [dt_us, dt_ms]}
    df_us = pd.DataFrame(arrays_us)
    schema_us = pa.schema(fields_us)
    filename = tempdir / 'pandas_truncated.parquet'
    table_us = pa.Table.from_pandas(df_us, schema=schema_us)
    _write_table(table_us, filename, version='2.6', coerce_timestamps='ms', allow_truncated_timestamps=True)
    table_ms = _read_table(filename)
    df_ms = table_ms.to_pandas()
    arrays_expected = {'datetime64': [dt_ms, dt_ms]}
    df_expected = pd.DataFrame(arrays_expected, dtype='datetime64[ms]')
    tm.assert_frame_equal(df_expected, df_ms)