import os
import shutil
import sys
import tempfile
import unittest
from io import StringIO
from pecan.tests import PecanTestCase
def test_destination_directory_already_exists(self):
    from pecan.scaffolds import copy_dir
    f = StringIO()
    copy_dir(('pecan', os.path.join('tests', 'scaffold_fixtures', 'simple')), os.path.join(self.scaffold_destination), {}, out_=f)
    assert 'already exists' in f.getvalue()