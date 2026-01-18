import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
@pytest.mark.parametrize('filename, newname', [('foobar.nii', 'foobar_0xabc123.nii'), ('foobar.nii.gz', 'foobar_0xabc123.nii.gz')])
def test_hash_rename(filename, newname):
    new_name = hash_rename(filename, 'abc123')
    assert new_name == newname