import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def chmod_file(self, path, mode, base=branch_dir):
    full_path = osutils.pathjoin(base, path)
    os.chmod(full_path, mode)
    self.tree.commit('change file {} mode to {}'.format(path, oct(mode)))