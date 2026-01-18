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
def test_construct_from_list_of_mixed_paths_fails(mockfs):
    files = ['subdir/1/xxx/file0.parquet', 'subdir/1/xxx/doesnt-exist.parquet']
    with pytest.raises(FileNotFoundError, match='doesnt-exist'):
        ds.dataset(files, filesystem=mockfs)