from hashlib import md5
from itertools import product
import pytest
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_copy_two_files_new_directory(self, fs, fs_join, fs_bulk_operations_scenario_0, fs_target):
    source = fs_bulk_operations_scenario_0
    target = fs_target
    assert not fs.exists(target)
    fs.cp([fs_join(source, 'file1'), fs_join(source, 'file2')], target)
    assert fs.isdir(target)
    assert fs.isfile(fs_join(target, 'file1'))
    assert fs.isfile(fs_join(target, 'file2'))