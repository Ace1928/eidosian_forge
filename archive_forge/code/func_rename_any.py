import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def rename_any(self, old_path, new_path):
    self.tree.rename_one(old_path, new_path)
    self.tree.commit('rename {} into {}'.format(old_path, new_path))