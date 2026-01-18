from breezy import branchbuilder, tests, transport, workingtree
from breezy.tests import per_controldir, test_server
from breezy.transport import memory
def make_branch_builder(self, relpath, format=None):
    if format is None:
        format = self.workingtree_format.get_controldir_for_branch()
    builder = branchbuilder.BranchBuilder(self.get_transport(relpath), format=format)
    return builder