import os
import shutil
import sys
import tempfile
import unittest
from io import StringIO
from pecan.tests import PecanTestCase
def test_copy_dir(self):
    from pecan.scaffolds import PecanScaffold

    class SimpleScaffold(PecanScaffold):
        _scaffold_dir = ('pecan', os.path.join('tests', 'scaffold_fixtures', 'simple'))
    SimpleScaffold().copy_to(os.path.join(self.scaffold_destination, 'someapp'), out_=StringIO())
    assert os.path.isfile(os.path.join(self.scaffold_destination, 'someapp', 'foo'))
    assert os.path.isfile(os.path.join(self.scaffold_destination, 'someapp', 'bar', 'spam.txt'))
    with open(os.path.join(self.scaffold_destination, 'someapp', 'foo'), 'r') as f:
        assert f.read().strip() == 'YAR'