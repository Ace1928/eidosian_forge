import os
import shutil
from io import StringIO
from .. import bisect
from ..controldir import ControlDir
from . import TestCaseWithTransport, TestSkipped
def testLoad(self):
    """Test loading a log."""
    preloaded_log = open(os.path.join('.bzr', bisect.BISECT_INFO_PATH), 'w')
    preloaded_log.write('rev1 yes\nrev2 no\nrev3 yes\n')
    preloaded_log.close()
    bisect_log = bisect.BisectLog(self.tree.controldir)
    self.assertEqual(len(bisect_log._items), 3)
    self.assertEqual(bisect_log._items[0], (b'rev1', 'yes'))
    self.assertEqual(bisect_log._items[1], (b'rev2', 'no'))
    self.assertEqual(bisect_log._items[2], (b'rev3', 'yes'))