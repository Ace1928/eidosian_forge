import datetime
import io
import warnings
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.pandas
@pytest.mark.parametrize('pq_reader_method', ['ParquetFile', 'read_table'])
def test_coerce_int96_timestamp_overflow(pq_reader_method, tempdir):

    def get_table(pq_reader_method, filename, **kwargs):
        if pq_reader_method == 'ParquetFile':
            return pq.ParquetFile(filename, **kwargs).read()
        elif pq_reader_method == 'read_table':
            return pq.read_table(filename, **kwargs)
    oob_dts = [datetime.datetime(1000, 1, 1), datetime.datetime(2000, 1, 1), datetime.datetime(3000, 1, 1)]
    df = pd.DataFrame({'a': oob_dts})
    table = pa.table(df)
    filename = tempdir / 'test_round_trip_overflow.parquet'
    pq.write_table(table, filename, use_deprecated_int96_timestamps=True, version='1.0')
    tab_error = get_table(pq_reader_method, filename)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Discarding nonzero nanoseconds in conversion', UserWarning)
        assert tab_error['a'].to_pylist() != oob_dts
    tab_correct = get_table(pq_reader_method, filename, coerce_int96_timestamp_unit='s')
    df_correct = tab_correct.to_pandas(timestamp_as_object=True)
    tm.assert_frame_equal(df, df_correct)