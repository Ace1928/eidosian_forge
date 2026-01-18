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
def test_fragments_parquet_pickle_no_metadata(tempdir, open_logging_fs, pickle_module):
    fs, assert_opens = open_logging_fs
    _, dataset = _create_dataset_for_fragments(tempdir, filesystem=fs)
    fragment = list(dataset.get_fragments())[1]
    with assert_opens([]):
        pickled_fragment = pickle_module.loads(pickle_module.dumps(fragment))
    with assert_opens([pickled_fragment.path]):
        row_groups = pickled_fragment.row_groups
    assert row_groups == [0]