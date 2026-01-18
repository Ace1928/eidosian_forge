import unittest
import os
from distutils.dep_util import newer, newer_pairwise, newer_group
from distutils.errors import DistutilsFileError
from distutils.tests import support
def test_newer_group(self):
    tmpdir = self.mkdtemp()
    sources = os.path.join(tmpdir, 'sources')
    os.mkdir(sources)
    one = os.path.join(sources, 'one')
    two = os.path.join(sources, 'two')
    three = os.path.join(sources, 'three')
    old_file = os.path.abspath(__file__)
    self.write_file(one)
    self.write_file(two)
    self.write_file(three)
    self.assertTrue(newer_group([one, two, three], old_file))
    self.assertFalse(newer_group([one, two, old_file], three))
    os.remove(one)
    self.assertRaises(OSError, newer_group, [one, two, old_file], three)
    self.assertFalse(newer_group([one, two, old_file], three, missing='ignore'))
    self.assertTrue(newer_group([one, two, old_file], three, missing='newer'))