import os
import pytest
def compare_paths(path1, path2):
    assert os.path.abspath(path1) == os.path.abspath(path2)