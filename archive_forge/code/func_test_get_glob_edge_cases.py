from hashlib import md5
from itertools import product
import pytest
from fsspec.implementations.local import make_path_posix
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
@pytest.mark.parametrize(GLOB_EDGE_CASES_TESTS['argnames'], GLOB_EDGE_CASES_TESTS['argvalues'])
def test_get_glob_edge_cases(self, path, recursive, maxdepth, expected, fs, fs_join, fs_glob_edge_cases_files, local_fs, local_join, local_target):
    source = fs_glob_edge_cases_files
    target = local_target
    for new_dir, target_slash in product([True, False], [True, False]):
        local_fs.mkdir(target)
        t = local_join(target, 'newdir') if new_dir else target
        t = t + '/' if target_slash else t
        fs.get(fs_join(source, path), t, recursive=recursive, maxdepth=maxdepth)
        output = local_fs.find(target)
        if new_dir:
            prefixed_expected = [make_path_posix(local_join(target, 'newdir', p)) for p in expected]
        else:
            prefixed_expected = [make_path_posix(local_join(target, p)) for p in expected]
        assert sorted(output) == sorted(prefixed_expected)
        try:
            local_fs.rm(target, recursive=True)
        except FileNotFoundError:
            pass