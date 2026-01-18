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
def py_fsspec_memoryfs(request, tempdir):
    fsspec = pytest.importorskip('fsspec', minversion='0.7.5')
    if fsspec.__version__ == '0.8.5':
        pytest.skip('Bug in fsspec 0.8.5 for in-memory filesystem')
    fs = fsspec.filesystem('memory')
    return dict(fs=PyFileSystem(FSSpecHandler(fs)), pathfn=lambda p: p, allow_move_dir=True, allow_append_to_file=True)