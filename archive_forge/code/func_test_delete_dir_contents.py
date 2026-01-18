from datetime import datetime, timezone, timedelta
import gzip
import os
import pathlib
import subprocess
import sys
import pytest
import weakref
import pyarrow as pa
from pyarrow.tests.test_io import assert_file_not_found
from pyarrow.tests.util import (_filesystem_uri, ProxyHandler,
from pyarrow.fs import (FileType, FileInfo, FileSelector, FileSystem,
def test_delete_dir_contents(fs, pathfn):
    skip_fsspec_s3fs(fs)
    d = pathfn('directory/')
    nd = pathfn('directory/nested/')
    fs.create_dir(nd)
    fs.delete_dir_contents(d)
    with pytest.raises(pa.ArrowIOError):
        fs.delete_dir(nd)
    fs.delete_dir_contents(nd, missing_dir_ok=True)
    with pytest.raises(pa.ArrowIOError):
        fs.delete_dir_contents(nd)
    fs.delete_dir(d)
    with pytest.raises(pa.ArrowIOError):
        fs.delete_dir(d)