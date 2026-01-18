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
def test_dataset_project_columns(tempdir, dataset_reader):
    from pyarrow import feather
    table = pa.table({'A': [1, 2, 3], 'B': [1.0, 2.0, 3.0], 'C': ['a', 'b', 'c']})
    feather.write_feather(table, tempdir / 'data.feather')
    dataset = ds.dataset(tempdir / 'data.feather', format='feather')
    result = dataset_reader.to_table(dataset, columns={'A_renamed': ds.field('A'), 'B_as_int': ds.field('B').cast('int32', safe=False), 'C_is_a': ds.field('C') == 'a'})
    expected = pa.table({'A_renamed': [1, 2, 3], 'B_as_int': pa.array([1, 2, 3], type='int32'), 'C_is_a': [True, False, False]})
    assert result.equals(expected)
    with pytest.raises(TypeError, match='Expected an Expression'):
        dataset_reader.to_table(dataset, columns={'A': 'A'})