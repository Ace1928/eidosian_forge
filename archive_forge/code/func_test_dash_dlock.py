import os
import signal
import sys
import time
from breezy import debug, tests
def test_dash_dlock(self):
    self.run_bzr('-Dlock init foo')
    self.assertContainsRe(self.get_log(), 'lock_write')