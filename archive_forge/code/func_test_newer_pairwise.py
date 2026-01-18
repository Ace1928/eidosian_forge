import unittest
import os
from distutils.dep_util import newer, newer_pairwise, newer_group
from distutils.errors import DistutilsFileError
from distutils.tests import support
def test_newer_pairwise(self):
    tmpdir = self.mkdtemp()
    sources = os.path.join(tmpdir, 'sources')
    targets = os.path.join(tmpdir, 'targets')
    os.mkdir(sources)
    os.mkdir(targets)
    one = os.path.join(sources, 'one')
    two = os.path.join(sources, 'two')
    three = os.path.abspath(__file__)
    four = os.path.join(targets, 'four')
    self.write_file(one)
    self.write_file(two)
    self.write_file(four)
    self.assertEqual(newer_pairwise([one, two], [three, four]), ([one], [three]))