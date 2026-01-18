from hashlib import md5
from itertools import product
import pytest
from fsspec.implementations.local import make_path_posix
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_get_directory_without_files_with_same_name_prefix(self, fs, fs_join, local_fs, local_join, local_target, fs_dir_and_file_with_same_name_prefix):
    source = fs_dir_and_file_with_same_name_prefix
    target = local_target
    fs.get(fs_join(source, 'subdir'), target, recursive=True)
    assert local_fs.isfile(local_join(target, 'subfile.txt'))
    assert not local_fs.isfile(local_join(target, 'subdir.txt'))
    local_fs.rm([local_join(target, 'subfile.txt')])
    assert local_fs.ls(target) == []
    fs.get(fs_join(source, 'subdir*'), target, recursive=True)
    assert local_fs.isdir(local_join(target, 'subdir'))
    assert local_fs.isfile(local_join(target, 'subdir', 'subfile.txt'))
    assert local_fs.isfile(local_join(target, 'subdir.txt'))