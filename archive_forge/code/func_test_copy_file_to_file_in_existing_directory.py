from hashlib import md5
from itertools import product
import pytest
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_copy_file_to_file_in_existing_directory(self, fs, fs_join, fs_bulk_operations_scenario_0, fs_target, supports_empty_directories):
    source = fs_bulk_operations_scenario_0
    target = fs_target
    fs.mkdir(target)
    if not supports_empty_directories:
        fs.touch(fs_join(target, 'dummy'))
    assert fs.isdir(target)
    fs.cp(fs_join(source, 'subdir', 'subfile1'), fs_join(target, 'newfile'))
    assert fs.isfile(fs_join(target, 'newfile'))