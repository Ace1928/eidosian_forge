import os
import shutil
import sys
import tempfile
import unittest
from io import StringIO
from pecan.tests import PecanTestCase
def test_copy_dir_with_file_content_substitution(self):
    from pecan.scaffolds import copy_dir
    copy_dir(('pecan', os.path.join('tests', 'scaffold_fixtures', 'content_sub')), os.path.join(self.scaffold_destination, 'someapp'), {'package': 'thingy'}, out_=StringIO())
    assert os.path.isfile(os.path.join(self.scaffold_destination, 'someapp', 'foo'))
    assert os.path.isfile(os.path.join(self.scaffold_destination, 'someapp', 'bar', 'spam.txt'))
    with open(os.path.join(self.scaffold_destination, 'someapp', 'foo'), 'r') as f:
        assert f.read().strip() == 'YAR thingy'
    with open(os.path.join(self.scaffold_destination, 'someapp', 'bar', 'spam.txt'), 'r') as f:
        assert f.read().strip() == 'Pecan thingy'