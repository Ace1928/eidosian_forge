import os
import platform
import shutil
import tempfile
import unittest
from gzip import GzipFile
from pathlib import Path
import pytest
from monty.shutil import (
def test_handle_sub_dirs(self):
    sub_dir = os.path.join(test_dir, 'gzip_dir', 'sub_dir')
    sub_file = os.path.join(sub_dir, 'new_tempfile')
    os.mkdir(sub_dir)
    with open(sub_file, 'w') as f:
        f.write('anotherwhat')
    gzip_dir(os.path.join(test_dir, 'gzip_dir'))
    assert os.path.exists(f'{sub_file}.gz')
    assert not os.path.exists(sub_file)
    with GzipFile(f'{sub_file}.gz') as g:
        assert g.readline().decode('utf-8') == 'anotherwhat'