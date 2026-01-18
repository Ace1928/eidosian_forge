import os
import shutil
import sys
import tempfile
import unittest
from io import StringIO
from pecan.tests import PecanTestCase
def test_copy_dir_with_filename_substitution(self):
    from pecan.scaffolds import copy_dir
    copy_dir(('pecan', os.path.join('tests', 'scaffold_fixtures', 'file_sub')), os.path.join(self.scaffold_destination, 'someapp'), {'package': 'thingy'}, out_=StringIO())
    assert os.path.isfile(os.path.join(self.scaffold_destination, 'someapp', 'foo_thingy'))
    assert os.path.isfile(os.path.join(self.scaffold_destination, 'someapp', 'bar_thingy', 'spam.txt'))
    with open(os.path.join(self.scaffold_destination, 'someapp', 'foo_thingy'), 'r') as f:
        assert f.read().strip() == 'YAR'
    with open(os.path.join(self.scaffold_destination, 'someapp', 'bar_thingy', 'spam.txt'), 'r') as f:
        assert f.read().strip() == 'Pecan'