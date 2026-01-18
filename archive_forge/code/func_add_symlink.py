import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def add_symlink(self, path, target, base=branch_dir):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    os.symlink(target, osutils.pathjoin(base, path))
    self.tree.add(path)
    self.tree.commit('add symlink {} -> {}'.format(path, target))