import io
import numpy as np
import pyarrow as pa
from pyarrow.tests import util
def make_sample_file(table_or_df):
    import pyarrow.parquet as pq
    if isinstance(table_or_df, pa.Table):
        a_table = table_or_df
    else:
        a_table = pa.Table.from_pandas(table_or_df)
    buf = io.BytesIO()
    _write_table(a_table, buf, compression='SNAPPY', version='2.6')
    buf.seek(0)
    return pq.ParquetFile(buf)