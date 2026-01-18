import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
@pytest.mark.pandas
@pytest.mark.parquet
def test_fragments_parquet_row_groups_dictionary(tempdir, dataset_reader):
    df = pd.DataFrame(dict(col1=['a', 'b'], col2=[1, 2]))
    df['col1'] = df['col1'].astype('category')
    pq.write_table(pa.table(df), tempdir / 'test_filter_dictionary.parquet')
    import pyarrow.dataset as ds
    dataset = ds.dataset(tempdir / 'test_filter_dictionary.parquet')
    result = dataset_reader.to_table(dataset, filter=ds.field('col1') == 'a')
    assert (df.iloc[0] == result.to_pandas()).all().all()