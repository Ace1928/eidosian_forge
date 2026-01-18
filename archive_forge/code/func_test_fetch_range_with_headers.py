from __future__ import annotations
import os
import subprocess
import sys
import time
import fsspec
import pytest
from fsspec.core import open_files
from packaging.version import parse as parse_version
import dask.bag as db
from dask.utils import tmpdir
def test_fetch_range_with_headers(dir_server):
    root = 'http://localhost:8999/'
    fn = files[0]
    headers = {'Date': 'Wed, 21 Oct 2015 07:28:00 GMT'}
    f = open_files(root + fn, headers=headers)[0]
    with f as f:
        data = f.read(length=1) + f.read(length=-1)
    with open(os.path.join(dir_server, fn), 'rb') as expected:
        assert data == expected.read()