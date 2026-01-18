import shutil
import sys
from breezy import (branch, controldir, errors, info, osutils, tests, upgrade,
from breezy.bzr import bzrdir
from breezy.transport import memory
def test_info_colocated(self):
    br = self.make_branch_and_tree('target', format='development-colo')
    target = br.controldir.create_branch(name='dichtbij')
    br.controldir.set_branch_reference(target)
    out, err = self.run_bzr('info target')
    self.assertEqual(out, 'Standalone tree (format: development-colo)\nLocation:\n            light checkout root: target\n  checkout of co-located branch: dichtbij\n')
    self.assertEqual(err, '')