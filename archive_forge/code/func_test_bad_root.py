import os
import shutil
import unittest
from monty.tempfile import ScratchDir
def test_bad_root(self):
    with ScratchDir('bad_groot') as d:
        assert d == test_dir