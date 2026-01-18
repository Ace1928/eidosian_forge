import pathlib
import platform
import sys
import pytest
from ..paths import file_uri_to_path, is_relative, normalized_uri
@pytest.mark.skipif(WIN, reason="can't test POSIX paths on Windows")
@pytest.mark.parametrize('file_uri, expected_posix_path', [['file:///C:/Windows/System32/Drivers/etc', '/C:/Windows/System32/Drivers/etc'], ['file:///C:/some%20dir/some%20file.txt', '/C:/some dir/some file.txt'], ['file:///home/user/some%20file.txt', '/home/user/some file.txt']])
def test_file_uri_to_path_posix(file_uri, expected_posix_path):
    assert file_uri_to_path(file_uri) == expected_posix_path