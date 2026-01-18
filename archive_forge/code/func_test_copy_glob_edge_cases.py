from hashlib import md5
from itertools import product
import pytest
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
@pytest.mark.parametrize(GLOB_EDGE_CASES_TESTS['argnames'], GLOB_EDGE_CASES_TESTS['argvalues'])
def test_copy_glob_edge_cases(self, path, recursive, maxdepth, expected, fs, fs_join, fs_glob_edge_cases_files, fs_target, fs_sanitize_path):
    source = fs_glob_edge_cases_files
    target = fs_target
    for new_dir, target_slash in product([True, False], [True, False]):
        fs.mkdir(target)
        t = fs_join(target, 'newdir') if new_dir else target
        t = t + '/' if target_slash else t
        fs.copy(fs_join(source, path), t, recursive=recursive, maxdepth=maxdepth)
        output = fs.find(target)
        if new_dir:
            prefixed_expected = [fs_sanitize_path(fs_join(target, 'newdir', p)) for p in expected]
        else:
            prefixed_expected = [fs_sanitize_path(fs_join(target, p)) for p in expected]
        assert sorted(output) == sorted(prefixed_expected)
        try:
            fs.rm(target, recursive=True)
        except FileNotFoundError:
            pass