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
def test_get_file_info(fs, pathfn):
    aaa = pathfn('a/aa/aaa/')
    bb = pathfn('a/bb')
    c = pathfn('c.txt')
    zzz = pathfn('zzz')
    fs.create_dir(aaa)
    with fs.open_output_stream(bb):
        pass
    with fs.open_output_stream(c) as fp:
        fp.write(b'test')
    aaa_info, bb_info, c_info, zzz_info = fs.get_file_info([aaa, bb, c, zzz])
    assert aaa_info.path == aaa
    assert 'aaa' in repr(aaa_info)
    assert aaa_info.extension == ''
    if fs.type_name == "py::fsspec+('s3', 's3a')":
        assert aaa_info.type == FileType.NotFound
    else:
        assert aaa_info.type == FileType.Directory
        assert 'FileType.Directory' in repr(aaa_info)
    assert aaa_info.size is None
    check_mtime_or_absent(aaa_info)
    assert bb_info.path == str(bb)
    assert bb_info.base_name == 'bb'
    assert bb_info.extension == ''
    assert bb_info.type == FileType.File
    assert 'FileType.File' in repr(bb_info)
    assert bb_info.size == 0
    if fs.type_name not in ['py::fsspec+memory', "py::fsspec+('s3', 's3a')"]:
        check_mtime(bb_info)
    assert c_info.path == str(c)
    assert c_info.base_name == 'c.txt'
    assert c_info.extension == 'txt'
    assert c_info.type == FileType.File
    assert 'FileType.File' in repr(c_info)
    assert c_info.size == 4
    if fs.type_name not in ['py::fsspec+memory', "py::fsspec+('s3', 's3a')"]:
        check_mtime(c_info)
    assert zzz_info.path == str(zzz)
    assert zzz_info.base_name == 'zzz'
    assert zzz_info.extension == ''
    assert zzz_info.type == FileType.NotFound
    assert zzz_info.size is None
    assert zzz_info.mtime is None
    assert 'FileType.NotFound' in repr(zzz_info)
    check_mtime_absent(zzz_info)
    aaa_info2 = fs.get_file_info(aaa)
    assert aaa_info.path == aaa_info2.path
    assert aaa_info.type == aaa_info2.type