import os
import shutil
from io import StringIO
from .. import bisect
from ..controldir import ControlDir
from . import TestCaseWithTransport, TestSkipped
def testIsMergePoint(self):
    """Test merge point detection."""
    current = bisect.BisectCurrent(self.tree.controldir)
    self.assertRevno(5)
    self.assertFalse(current.is_merge_point())
    current.switch(2)
    self.assertTrue(current.is_merge_point())