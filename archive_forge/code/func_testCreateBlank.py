import os
import shutil
from io import StringIO
from .. import bisect
from ..controldir import ControlDir
from . import TestCaseWithTransport, TestSkipped
def testCreateBlank(self):
    """Test creation of new log."""
    bisect_log = bisect.BisectLog(self.tree.controldir)
    bisect_log.save()
    self.assertTrue(os.path.exists(os.path.join('.bzr', bisect.BISECT_INFO_PATH)))