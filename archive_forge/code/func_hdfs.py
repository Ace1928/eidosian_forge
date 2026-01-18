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
@pytest.fixture
def hdfs(request, hdfs_connection):
    request.config.pyarrow.requires('hdfs')
    if not pa.have_libhdfs():
        pytest.skip('Cannot locate libhdfs')
    from pyarrow.fs import HadoopFileSystem
    host, port, user = hdfs_connection
    fs = HadoopFileSystem(host, port=port, user=user)
    return dict(fs=fs, pathfn=lambda p: p, allow_move_dir=True, allow_append_to_file=True)