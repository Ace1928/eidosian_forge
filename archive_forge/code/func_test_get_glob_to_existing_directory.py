from hashlib import md5
from itertools import product
import pytest
from fsspec.implementations.local import make_path_posix
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_get_glob_to_existing_directory(self, fs, fs_join, fs_bulk_operations_scenario_0, local_fs, local_join, local_target):
    source = fs_bulk_operations_scenario_0
    target = local_target
    local_fs.mkdir(target)
    for target_slash in [False, True]:
        t = target + '/' if target_slash else target
        fs.get(fs_join(source, 'subdir', '*'), t)
        assert local_fs.isfile(local_join(target, 'subfile1'))
        assert local_fs.isfile(local_join(target, 'subfile2'))
        assert not local_fs.isdir(local_join(target, 'nesteddir'))
        assert not local_fs.exists(local_join(target, 'nesteddir', 'nestedfile'))
        assert not local_fs.exists(local_join(target, 'subdir'))
        local_fs.rm([local_join(target, 'subfile1'), local_join(target, 'subfile2')], recursive=True)
        assert local_fs.ls(target) == []
        for glob, recursive in zip(['*', '**'], [True, False]):
            fs.get(fs_join(source, 'subdir', glob), t, recursive=recursive)
            assert local_fs.isfile(local_join(target, 'subfile1'))
            assert local_fs.isfile(local_join(target, 'subfile2'))
            assert local_fs.isdir(local_join(target, 'nesteddir'))
            assert local_fs.isfile(local_join(target, 'nesteddir', 'nestedfile'))
            assert not local_fs.exists(local_join(target, 'subdir'))
            local_fs.rm([local_join(target, 'subfile1'), local_join(target, 'subfile2'), local_join(target, 'nesteddir')], recursive=True)
            assert local_fs.ls(target) == []
            fs.get(fs_join(source, 'subdir', glob), t, recursive=recursive, maxdepth=1)
            assert local_fs.isfile(local_join(target, 'subfile1'))
            assert local_fs.isfile(local_join(target, 'subfile2'))
            assert not local_fs.exists(local_join(target, 'nesteddir'))
            assert not local_fs.exists(local_join(target, 'subdir'))
            local_fs.rm([local_join(target, 'subfile1'), local_join(target, 'subfile2')], recursive=True)
            assert local_fs.ls(target) == []