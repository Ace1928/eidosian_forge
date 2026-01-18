import os
import shutil
from io import StringIO
from .. import bisect
from ..controldir import ControlDir
from . import TestCaseWithTransport, TestSkipped
def testShowLogSubtree(self):
    """Test that a subtree's log can be shown."""
    current = bisect.BisectCurrent(self.tree.controldir)
    current.switch(self.subtree_rev)
    sio = StringIO()
    current.show_rev_log(outf=sio)