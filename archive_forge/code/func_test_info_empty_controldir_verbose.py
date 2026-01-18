import shutil
import sys
from breezy import (branch, controldir, errors, info, osutils, tests, upgrade,
from breezy.bzr import bzrdir
from breezy.transport import memory
def test_info_empty_controldir_verbose(self):
    self.make_controldir('ctrl')
    out, err = self.run_bzr('info -v ctrl')
    self.assertEqualDiff(out, 'Empty control directory (format: 2a)\nLocation:\n  control directory: ctrl\n\nFormat:\n       control: Meta directory format 1\n\nControl directory:\n         0 branches\n')
    self.assertEqual(err, '')