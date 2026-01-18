import shutil
import sys
from breezy import (branch, controldir, errors, info, osutils, tests, upgrade,
from breezy.bzr import bzrdir
from breezy.transport import memory
def test_info_stacked(self):
    trunk_tree = self.make_branch_and_tree('mainline', format='1.6')
    trunk_tree.commit('mainline')
    new_dir = trunk_tree.controldir.sprout('newbranch', stacked=True)
    out, err = self.run_bzr('info newbranch')
    self.assertEqual('Standalone tree (format: 1.6)\nLocation:\n  branch root: newbranch\n\nRelated branches:\n  parent branch: mainline\n     stacked on: mainline\n', out)
    self.assertEqual('', err)