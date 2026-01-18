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
@pytest.mark.dataset
def test_dataset_join_unique_key(tempdir):
    t1 = pa.table({'colA': [1, 2, 6], 'col2': ['a', 'b', 'f']})
    ds.write_dataset(t1, tempdir / 't1', format='ipc')
    ds1 = ds.dataset(tempdir / 't1', format='ipc')
    t2 = pa.table({'colA': [99, 2, 1], 'col3': ['Z', 'B', 'A']})
    ds.write_dataset(t2, tempdir / 't2', format='ipc')
    ds2 = ds.dataset(tempdir / 't2', format='ipc')
    result = ds1.join(ds2, 'colA')
    assert result.to_table() == pa.table({'colA': [1, 2, 6], 'col2': ['a', 'b', 'f'], 'col3': ['A', 'B', None]})
    result = ds1.join(ds2, 'colA', join_type='full outer', right_suffix='_r')
    assert result.to_table().sort_by('colA') == pa.table({'colA': [1, 2, 6, 99], 'col2': ['a', 'b', 'f', None], 'col3': ['A', 'B', None, 'Z']})