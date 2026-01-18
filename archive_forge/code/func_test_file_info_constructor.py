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
def test_file_info_constructor():
    dt = datetime.fromtimestamp(1568799826, timezone.utc)
    info = FileInfo('foo/bar')
    assert info.path == 'foo/bar'
    assert info.base_name == 'bar'
    assert info.type == FileType.Unknown
    assert info.size is None
    check_mtime_absent(info)
    info = FileInfo('foo/baz.txt', type=FileType.File, size=123, mtime=1568799826.5)
    assert info.path == 'foo/baz.txt'
    assert info.base_name == 'baz.txt'
    assert info.type == FileType.File
    assert info.size == 123
    assert info.mtime_ns == 1568799826500000000
    check_mtime(info)
    info = FileInfo('foo', type=FileType.Directory, mtime=dt)
    assert info.path == 'foo'
    assert info.base_name == 'foo'
    assert info.type == FileType.Directory
    assert info.size is None
    assert info.mtime == dt
    assert info.mtime_ns == 1568799826000000000
    check_mtime(info)