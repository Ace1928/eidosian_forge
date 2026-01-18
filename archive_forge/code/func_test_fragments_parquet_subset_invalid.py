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
def test_fragments_parquet_subset_invalid(tempdir):
    _, dataset = _create_dataset_for_fragments(tempdir, chunk_size=1)
    fragment = list(dataset.get_fragments())[0]
    with pytest.raises(ValueError):
        fragment.subset(ds.field('f1') >= 1, row_group_ids=[1, 2])
    with pytest.raises(ValueError):
        fragment.subset()