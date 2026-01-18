import os
import shutil
import sys
import tempfile
import unittest
from io import StringIO
from pecan.tests import PecanTestCase
def test_destination_directory_levels_deep(self):
    from pecan.scaffolds import copy_dir
    f = StringIO()
    copy_dir(('pecan', os.path.join('tests', 'scaffold_fixtures', 'simple')), os.path.join(self.scaffold_destination, 'some', 'app'), {}, out_=f)
    assert os.path.isfile(os.path.join(self.scaffold_destination, 'some', 'app', 'foo'))
    assert os.path.isfile(os.path.join(self.scaffold_destination, 'some', 'app', 'bar', 'spam.txt'))
    with open(os.path.join(self.scaffold_destination, 'some', 'app', 'foo'), 'r') as f:
        assert f.read().strip() == 'YAR'
    with open(os.path.join(self.scaffold_destination, 'some', 'app', 'bar', 'spam.txt'), 'r') as f:
        assert f.read().strip() == 'Pecan'