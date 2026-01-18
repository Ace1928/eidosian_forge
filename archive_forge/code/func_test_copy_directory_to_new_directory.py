from hashlib import md5
from itertools import product
import pytest
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_copy_directory_to_new_directory(self, fs, fs_join, fs_bulk_operations_scenario_0, fs_target, supports_empty_directories):
    source = fs_bulk_operations_scenario_0
    target = fs_target
    fs.mkdir(target)
    for source_slash, target_slash in zip([False, True], [False, True]):
        s = fs_join(source, 'subdir')
        if source_slash:
            s += '/'
        t = fs_join(target, 'newdir')
        if target_slash:
            t += '/'
        fs.cp(s, t)
        if supports_empty_directories:
            assert fs.ls(target) == []
        else:
            with pytest.raises(FileNotFoundError):
                fs.ls(target)
        fs.cp(s, t, recursive=True)
        assert fs.isdir(fs_join(target, 'newdir'))
        assert fs.isfile(fs_join(target, 'newdir', 'subfile1'))
        assert fs.isfile(fs_join(target, 'newdir', 'subfile2'))
        assert fs.isdir(fs_join(target, 'newdir', 'nesteddir'))
        assert fs.isfile(fs_join(target, 'newdir', 'nesteddir', 'nestedfile'))
        assert not fs.exists(fs_join(target, 'subdir'))
        fs.rm(fs_join(target, 'newdir'), recursive=True)
        assert not fs.exists(fs_join(target, 'newdir'))
        fs.cp(s, t, recursive=True, maxdepth=1)
        assert fs.isdir(fs_join(target, 'newdir'))
        assert fs.isfile(fs_join(target, 'newdir', 'subfile1'))
        assert fs.isfile(fs_join(target, 'newdir', 'subfile2'))
        assert not fs.exists(fs_join(target, 'newdir', 'nesteddir'))
        assert not fs.exists(fs_join(target, 'subdir'))
        fs.rm(fs_join(target, 'newdir'), recursive=True)
        assert not fs.exists(fs_join(target, 'newdir'))