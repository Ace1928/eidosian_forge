from hashlib import md5
from itertools import product
import pytest
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_copy_with_source_and_destination_as_list(self, fs, fs_target, fs_join, local_join, local_10_files_with_hashed_names):
    source = local_10_files_with_hashed_names
    target = fs_target
    source_files = []
    destination_files = []
    for i in range(10):
        hashed_i = md5(str(i).encode('utf-8')).hexdigest()
        source_files.append(local_join(source, f'{hashed_i}.txt'))
        destination_files.append(fs_join(target, f'{hashed_i}.txt'))
    fs.put(lpath=source_files, rpath=destination_files)
    for i in range(10):
        file_content = fs.cat(destination_files[i]).decode('utf-8')
        assert file_content == str(i)