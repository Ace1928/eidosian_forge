import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def create_trunk_and_feature_branch(self):
    trunk_tree = self.make_branch_and_tree('target', format='1.9')
    trunk_tree.commit('mainline')
    branch_tree = self.make_branch_and_tree('branch', format='1.9')
    branch_tree.pull(trunk_tree.branch)
    branch_tree.branch.set_parent(trunk_tree.branch.base)
    branch_tree.commit('moar work plz')
    return (trunk_tree, branch_tree)