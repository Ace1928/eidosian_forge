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
def test_filesystem_is_functional_after_pickling(fs, pathfn, pickle_module):
    if fs.type_name.split('::')[-1] == 'mock':
        pytest.xfail(reason='MockFileSystem is not serializable')
    skip_fsspec_s3fs(fs)
    aaa = pathfn('a/aa/aaa/')
    bb = pathfn('a/bb')
    c = pathfn('c.txt')
    fs.create_dir(aaa)
    with fs.open_output_stream(bb):
        pass
    with fs.open_output_stream(c) as fp:
        fp.write(b'test')
    restored = pickle_module.loads(pickle_module.dumps(fs))
    aaa_info, bb_info, c_info = restored.get_file_info([aaa, bb, c])
    assert aaa_info.type == FileType.Directory
    assert bb_info.type == FileType.File
    assert c_info.type == FileType.File