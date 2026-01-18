import shutil
import sys
from breezy import (branch, controldir, errors, info, osutils, tests, upgrade,
from breezy.bzr import bzrdir
from breezy.transport import memory
def test_info_empty_controldir(self):
    self.make_controldir('ctrl')
    out, err = self.run_bzr('info ctrl')
    self.assertEqual(out, 'Empty control directory (format: 2a)\nLocation:\n  control directory: ctrl\n')
    self.assertEqual(err, '')