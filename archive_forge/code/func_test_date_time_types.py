import datetime
import io
import warnings
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.pandas
def test_date_time_types(tempdir):
    t1 = pa.date32()
    data1 = np.array([17259, 17260, 17261], dtype='int32')
    a1 = pa.array(data1, type=t1)
    t2 = pa.date64()
    data2 = data1.astype('int64') * 86400000
    a2 = pa.array(data2, type=t2)
    t3 = pa.timestamp('us')
    start = pd.Timestamp('2001-01-01').value / 1000
    data3 = np.array([start, start + 1, start + 2], dtype='int64')
    a3 = pa.array(data3, type=t3)
    t4 = pa.time32('ms')
    data4 = np.arange(3, dtype='i4')
    a4 = pa.array(data4, type=t4)
    t5 = pa.time64('us')
    a5 = pa.array(data4.astype('int64'), type=t5)
    t6 = pa.time32('s')
    a6 = pa.array(data4, type=t6)
    ex_t6 = pa.time32('ms')
    ex_a6 = pa.array(data4 * 1000, type=ex_t6)
    t7 = pa.timestamp('ns')
    start = pd.Timestamp('2001-01-01').value
    data7 = np.array([start, start + 1000, start + 2000], dtype='int64')
    a7 = pa.array(data7, type=t7)
    table = pa.Table.from_arrays([a1, a2, a3, a4, a5, a6, a7], ['date32', 'date64', 'timestamp[us]', 'time32[s]', 'time64[us]', 'time32_from64[s]', 'timestamp[ns]'])
    expected = pa.Table.from_arrays([a1, a1, a3, a4, a5, ex_a6, a7], ['date32', 'date64', 'timestamp[us]', 'time32[s]', 'time64[us]', 'time32_from64[s]', 'timestamp[ns]'])
    _check_roundtrip(table, expected=expected, version='2.6')
    t0 = pa.timestamp('ms')
    data0 = np.arange(4, dtype='int64')
    a0 = pa.array(data0, type=t0)
    t1 = pa.timestamp('us')
    data1 = np.arange(4, dtype='int64')
    a1 = pa.array(data1, type=t1)
    t2 = pa.timestamp('ns')
    data2 = np.arange(4, dtype='int64')
    a2 = pa.array(data2, type=t2)
    table = pa.Table.from_arrays([a0, a1, a2], ['ts[ms]', 'ts[us]', 'ts[ns]'])
    expected = pa.Table.from_arrays([a0, a1, a2], ['ts[ms]', 'ts[us]', 'ts[ns]'])
    filename = tempdir / 'int64_timestamps.parquet'
    _write_table(table, filename, version='2.6')
    parquet_schema = pq.ParquetFile(filename).schema
    for i in range(3):
        assert parquet_schema.column(i).physical_type == 'INT64'
    read_table = _read_table(filename)
    assert read_table.equals(expected)
    t0_ns = pa.timestamp('ns')
    data0_ns = np.array(data0 * 1000000, dtype='int64')
    a0_ns = pa.array(data0_ns, type=t0_ns)
    t1_ns = pa.timestamp('ns')
    data1_ns = np.array(data1 * 1000, dtype='int64')
    a1_ns = pa.array(data1_ns, type=t1_ns)
    expected = pa.Table.from_arrays([a0_ns, a1_ns, a2], ['ts[ms]', 'ts[us]', 'ts[ns]'])
    filename = tempdir / 'explicit_int96_timestamps.parquet'
    _write_table(table, filename, version='2.6', use_deprecated_int96_timestamps=True)
    parquet_schema = pq.ParquetFile(filename).schema
    for i in range(3):
        assert parquet_schema.column(i).physical_type == 'INT96'
    read_table = _read_table(filename)
    assert read_table.equals(expected)
    filename = tempdir / 'spark_int96_timestamps.parquet'
    _write_table(table, filename, version='2.6', flavor='spark')
    parquet_schema = pq.ParquetFile(filename).schema
    for i in range(3):
        assert parquet_schema.column(i).physical_type == 'INT96'
    read_table = _read_table(filename)
    assert read_table.equals(expected)