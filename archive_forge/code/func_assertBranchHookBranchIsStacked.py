import os
from breezy import branch as _mod_branch
from breezy import errors, osutils
from breezy import revision as _mod_revision
from breezy import tests, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import remote
from breezy.tests import features
from breezy.tests.per_branch import TestCaseWithBranch
def assertBranchHookBranchIsStacked(self, pre_change_params):
    pre_change_params.branch.get_stacked_on_url()
    self.hook_calls.append(pre_change_params)