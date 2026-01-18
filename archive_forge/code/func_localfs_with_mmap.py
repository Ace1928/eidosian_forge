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
def localfs_with_mmap(request, tempdir):
    return dict(fs=LocalFileSystem(use_mmap=True), pathfn=lambda p: (tempdir / p).as_posix(), allow_move_dir=True, allow_append_to_file=True)