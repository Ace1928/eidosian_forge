from collections import OrderedDict
import io
import warnings
from shutil import copytree
import numpy as np
import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem, FileSystem
from pyarrow.tests import util
from pyarrow.tests.parquet.common import (_check_roundtrip, _roundtrip_table,
@pytest.mark.pandas
def test_chunked_table_write():
    tables = []
    batch = pa.RecordBatch.from_pandas(alltypes_sample(size=10))
    tables.append(pa.Table.from_batches([batch] * 3))
    df, _ = dataframe_with_lists()
    batch = pa.RecordBatch.from_pandas(df)
    tables.append(pa.Table.from_batches([batch] * 3))
    for data_page_version in ['1.0', '2.0']:
        for use_dictionary in [True, False]:
            for table in tables:
                _check_roundtrip(table, version='2.6', data_page_version=data_page_version, use_dictionary=use_dictionary)