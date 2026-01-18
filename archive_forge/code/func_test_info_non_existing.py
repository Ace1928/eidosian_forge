import shutil
import sys
from breezy import (branch, controldir, errors, info, osutils, tests, upgrade,
from breezy.bzr import bzrdir
from breezy.transport import memory
def test_info_non_existing(self):
    self.vfs_transport_factory = memory.MemoryServer
    location = self.get_url()
    out, err = self.run_bzr('info ' + location, retcode=3)
    self.assertEqual(out, '')
    self.assertEqual(err, 'brz: ERROR: Not a branch: "%s".\n' % location)