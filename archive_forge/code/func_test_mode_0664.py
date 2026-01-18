import os
import stat
import sys
from .. import atomicfile, osutils
from . import TestCaseInTempDir, TestSkipped
def test_mode_0664(self):
    self._test_mode(436)