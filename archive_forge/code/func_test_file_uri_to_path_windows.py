import pathlib
import platform
import sys
import pytest
from ..paths import file_uri_to_path, is_relative, normalized_uri
@pytest.mark.skipif(not WIN, reason="can't test Windows paths on POSIX")
@pytest.mark.parametrize('file_uri, expected_windows_path', [['file:///C:/Windows/System32/Drivers/etc', 'C:/Windows/System32/Drivers/etc'], ['file:///C:/some%20dir/some%20file.txt', 'C:/some dir/some file.txt']])
def test_file_uri_to_path_windows(file_uri, expected_windows_path):
    assert file_uri_to_path(file_uri) == expected_windows_path