import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def modify_file(self, path, content, base=branch_dir):
    self.set_file_content(path, content, base)
    self.tree.commit('modify file %s' % path)