from hashlib import md5
from itertools import product
import pytest
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_copy_list_of_files_to_existing_directory(self, fs, fs_join, fs_bulk_operations_scenario_0, fs_target, supports_empty_directories):
    source = fs_bulk_operations_scenario_0
    target = fs_target
    fs.mkdir(target)
    if not supports_empty_directories:
        dummy = fs_join(target, 'dummy')
        fs.touch(dummy)
    assert fs.isdir(target)
    source_files = [fs_join(source, 'file1'), fs_join(source, 'file2'), fs_join(source, 'subdir', 'subfile1')]
    for target_slash in [False, True]:
        t = target + '/' if target_slash else target
        fs.cp(source_files, t)
        assert fs.isfile(fs_join(target, 'file1'))
        assert fs.isfile(fs_join(target, 'file2'))
        assert fs.isfile(fs_join(target, 'subfile1'))
        fs.rm([fs_join(target, 'file1'), fs_join(target, 'file2'), fs_join(target, 'subfile1')], recursive=True)
        assert fs.ls(target, detail=False) == ([] if supports_empty_directories else [dummy])