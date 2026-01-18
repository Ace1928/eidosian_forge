import os
import pytest
from monty.os import cd, makedirs_p
from monty.os.path import find_exts, zpath
def test_cd_exception(self):
    with cd(test_dir):
        assert os.path.exists('empty_file.txt')
    assert not os.path.exists('empty_file.txt')