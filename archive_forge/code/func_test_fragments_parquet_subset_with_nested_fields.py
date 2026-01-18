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
@pytest.mark.parquet
def test_fragments_parquet_subset_with_nested_fields(tempdir):
    f1 = pa.array([0, 1, 2, 3])
    f21 = pa.array([0.1, 0.2, 0.3, 0.4])
    f22 = pa.array([1, 2, 3, 4])
    f2 = pa.StructArray.from_arrays([f21, f22], names=['f21', 'f22'])
    struct_col = pa.StructArray.from_arrays([f1, f2], names=['f1', 'f2'])
    table = pa.table({'col': struct_col})
    pq.write_table(table, tempdir / 'data_struct.parquet', row_group_size=2)
    dataset = ds.dataset(tempdir / 'data_struct.parquet', format='parquet')
    fragment = list(dataset.get_fragments())[0]
    assert fragment.num_row_groups == 2
    subfrag = fragment.subset(ds.field('col', 'f1') > 2)
    assert subfrag.num_row_groups == 1
    subfrag = fragment.subset(ds.field('col', 'f1') > 5)
    assert subfrag.num_row_groups == 0
    subfrag = fragment.subset(ds.field('col', 'f2', 'f21') > 0)
    assert subfrag.num_row_groups == 2
    subfrag = fragment.subset(ds.field('col', 'f2', 'f22') <= 2)
    assert subfrag.num_row_groups == 1
    with pytest.raises(pa.ArrowInvalid, match='No match for FieldRef.Nested'):
        fragment.subset(ds.field('col', 'f3') > 0)
    with pytest.raises(NotImplementedError, match="Function 'greater' has no kernel matching"):
        fragment.subset(ds.field('col', 'f2') > 0)