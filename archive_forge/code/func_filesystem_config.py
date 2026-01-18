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
@pytest.fixture(params=[pytest.param('localfs', id='LocalFileSystem()'), pytest.param('localfs_with_mmap', id='LocalFileSystem(use_mmap=True)'), pytest.param('subtree_localfs', id='SubTreeFileSystem(LocalFileSystem())'), pytest.param('s3fs', id='S3FileSystem', marks=pytest.mark.s3), pytest.param('gcsfs', id='GcsFileSystem', marks=pytest.mark.gcs), pytest.param('hdfs', id='HadoopFileSystem', marks=pytest.mark.hdfs), pytest.param('mockfs', id='_MockFileSystem()'), pytest.param('py_localfs', id='PyFileSystem(ProxyHandler(LocalFileSystem()))'), pytest.param('py_mockfs', id='PyFileSystem(ProxyHandler(_MockFileSystem()))'), pytest.param('py_fsspec_localfs', id='PyFileSystem(FSSpecHandler(fsspec.LocalFileSystem()))'), pytest.param('py_fsspec_memoryfs', id='PyFileSystem(FSSpecHandler(fsspec.filesystem("memory")))'), pytest.param('py_fsspec_s3fs', id='PyFileSystem(FSSpecHandler(s3fs.S3FileSystem()))', marks=pytest.mark.s3)])
def filesystem_config(request):
    return request.getfixturevalue(request.param)