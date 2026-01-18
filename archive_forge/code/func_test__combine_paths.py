import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test__combine_paths(self):
    combine = urlutils.URL._combine_paths
    self.assertEqual('/home/sarah/project/foo', combine('/home/sarah', 'project/foo'))
    self.assertEqual('/etc', combine('/home/sarah', '../../etc'))
    self.assertEqual('/etc', combine('/home/sarah', '../../../etc'))
    self.assertEqual('/etc', combine('/home/sarah', '/etc'))