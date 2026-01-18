import os
import unittest
import pytest
from monty.io import (
@unittest.skipIf(Path is None, 'Not Py3k')
def test_Path_objects(self):
    p = Path(test_dir) / 'myfile_gz.gz'
    with zopen(p, mode='rt') as f:
        assert f.read() == 'HelloWorld.\n\n'