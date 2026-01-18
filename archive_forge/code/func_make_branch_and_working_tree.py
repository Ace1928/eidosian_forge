import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def make_branch_and_working_tree(self):
    t = transport.get_transport(self.branch_dir)
    t.ensure_base()
    branch = controldir.ControlDir.create_branch_convenience(t.base, format=controldir.format_registry.make_controldir('default'), force_new_tree=False)
    self.tree = branch.controldir.create_workingtree()
    self.tree.commit('initial empty tree')