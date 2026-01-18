from __future__ import annotations
import io
import os
import pathlib
import pytest
from fsspec.utils import (
@pytest.mark.parametrize('urlpath, expected_path', (('c:\\foo\\bar', 'c:\\foo\\bar'), ('C:\\\\foo\\bar', 'C:\\\\foo\\bar'), ('c:/foo/bar', 'c:/foo/bar'), ('file:///c|\\foo\\bar', 'c:\\foo\\bar'), ('file:///C|/foo/bar', 'C:/foo/bar'), ('file:///C:/foo/bar', 'C:/foo/bar')))
def test_infer_storage_options_c(urlpath, expected_path):
    so = infer_storage_options(urlpath)
    assert so['protocol'] == 'file'
    assert so['path'] == expected_path