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
@pytest.mark.flaky(reruns=10, reruns_delay=5, reason='https://github.com/dask/dask/issues/3696')
@pytest.mark.network
def test_bag():
    urls = ['https://raw.githubusercontent.com/weierophinney/pastebin/master/public/js-src/dojox/data/tests/stores/patterns.csv', 'https://en.wikipedia.org']
    b = db.read_text(urls)
    assert b.npartitions == 2
    b.compute()