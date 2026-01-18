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
def test_empty_table_no_columns():
    df = pd.DataFrame()
    empty = pa.Table.from_pandas(df, preserve_index=False)
    _check_roundtrip(empty)